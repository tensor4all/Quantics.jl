function _extractsite(x::Union{MPS,MPO}, n::Int)
    if n == 1
        return noprime(copy(uniqueind(x[n], x[n + 1])))
    elseif n == length(x)
        return noprime(copy(uniqueind(x[n], x[n - 1])))
    else
        return noprime(copy(uniqueind(x[n], x[n + 1], x[n - 1])))
    end
end

_extractsites(x::Union{MPS,MPO}) = [_extractsite(x, n) for n in eachindex(x)]

function _replace_mpo_siteinds!(M::MPO, sites_src, sites_dst)
    sites_src = noprime(sites_src)
    sites_dst = noprime(sites_dst)
    for j in eachindex(M)
        replaceind!(M[j], sites_src[j], sites_dst[j])
        replaceind!(M[j], sites_src[j]', sites_dst[j]')
    end
    return M
end

"""
Reverse the order of the MPS/MPO tensors
The order of the siteinds are reversed in the returned object.
"""
function _reverse(M::MPO)
    sites = _extractsites(M)
    N = length(M)
    M_ = MPO([M[n] for n in reverse(1:length(M))])
    for n in 1:N
        replaceind!(M_[n], sites[N - n + 1], sites[n])
        replaceind!(M_[n], sites[N - n + 1]', sites[n]')
    end
    return M_
end

"""
Create a MPO with ITensor objects of ElType ComplexF64 filled with zero
"""
function _zero_mpo(sites; linkdims=ones(Int, length(sites) - 1))
    length(linkdims) == length(sites) - 1 ||
        error("Length mismatch $(length(linkdims)) != $(length(sites)) - 1")
    M = MPO(sites)
    N = length(M)
    links = [Index(1, "n=0,Link")]
    for n in 1:(N - 1)
        push!(links, Index(linkdims[n], "n=$(n),Link"))
    end
    push!(links, Index(1, "n=$N,Link"))
    for n in 1:N
        inds_ = (links[n], sites[n]', sites[n], links[n + 1])
        elm_ = zeros(ComplexF64, map(ITensors.dim, inds_)...)
        M[n] = ITensor(elm_, inds_...)
    end
    M[1] *= ITensors.delta(links[1])
    M[N] *= ITensors.delta(links[N + 1])

    return M
end

# Compute linkdims for a maximally entangled state
function maxlinkdims(inds)
    N = length(inds)
    for i in 1:N
        @assert !ITensors.hastags(inds, "Link")
    end

    physdims = dim.(inds)

    maxdim = ones(Float64, N - 1)
    maxdiml = 1.0
    for i in 1:(N - 1)
        maxdiml *= physdims[i]
        maxdim[i] = maxdiml
    end

    maxdimr = 1.0
    for i in 1:(N - 1)
        maxdimr *= physdims[N + 1 - i]
        maxdim[N - i] = min(maxdimr, maxdim[N - i])
    end
    return maxdim
end

"""
Un-fuse the site indices of an MPS at the given sites

M: Input MPS where each tensor has only one site index
target_sites: Vector of siteinds to be split
new_sites: Vector of vectors of new siteinds

When splitting MPS tensors, the column major is assumed.
"""
function unfuse_siteinds(M::MPS, targetsites::Vector{Index{T}},
        newsites::AbstractVector{Vector{Index{T}}})::MPS where {T}
    length(targetsites) == length(newsites) || error("Length mismatch")
    links = linkinds(M)
    L = length(M)

    tensors = Union{ITensor,Vector{ITensor}}[deepcopy(M[n]) for n in eachindex(M)]
    for n in 1:length(targetsites)
        pos = findsite(M, targetsites[n])
        !isnothing(pos) || error("Target site not found: $(targetsites[n])")

        newinds = [[s] for s in newsites[n]]
        links_ = Index{T}[]
        if pos > 1
            push!(links_, links[pos - 1])
            push!(newinds[1], links[pos - 1])
        end
        if pos < L
            push!(links_, links[pos])
            push!(newinds[end], links[pos])
        end
        tensor_data = ITensors.data(permute(copy(M[pos]), targetsites[n], links_...))
        tensors[pos] = split_tensor(ITensor(tensor_data, newsites[n]..., links_...),
            newinds)
    end

    tensors_ = ITensor[]
    for t in tensors
        if t isa ITensor
            push!(tensors_, t)
        elseif t isa Vector{ITensor}
            for t_ in t
                push!(tensors_, t_)
            end
        end
    end

    M_ = MPS(tensors_)
    cleanup_linkinds!(M_)
    return M_
end

function _removeedges!(x::MPS, sites)
    length(inds(x[1])) == 3 || error("Dim of the first tensor must be 3")
    length(inds(x[end])) == 3 || error("Dim of the last tensor must be 3")
    elt = eltype(x[1])
    x[1] *= onehot(elt, uniqueind(x[1], x[2], sites) => 1)
    x[end] *= onehot(elt, uniqueind(x[end], x[end - 1], sites) => 1)
    return nothing
end

function _removeedges!(x::MPO, sites)
    length(inds(x[1])) == 4 || error("Dim of the first tensor must be 4")
    length(inds(x[end])) == 4 || error("Dim of the last tensor must be 4")
    elt = eltype(x[1])
    x[1] *= onehot(elt, uniqueind(x[1], x[2], sites, prime.(sites)) => 1)
    x[end] *= onehot(elt, uniqueind(x[end], x[end - 1], sites, prime.(sites)) => 1)
    return nothing
end

function _removeedges!(tensors::Vector{ITensor}, sites)
    tensors[1] *= onehot(Float64,
        uniqueind(tensors[1], tensors[2], sites, prime.(sites)) => 1)
    tensors[end] *= onehot(Float64,
        uniqueind(tensors[end], tensors[end - 1], sites, prime.(sites)) => 1)
end

function _addedges!(x::MPS)
    length(inds(x[1])) == 2 || error("Dim of the first tensor must be 2")
    length(inds(x[end])) == 2 || error("Dim of the last tensor must be 2")
    linkl = Index(1, "Link,l=0")
    linkr = Index(1, "Link,l=$(length(x))")
    x[1] = ITensor(ITensors.data(x[1]), [linkl, inds(x[1])...])
    x[end] = ITensor(ITensors.data(x[end]), [inds(x[end])..., linkr])
    return nothing
end

function _addedges!(x::MPO)
    length(inds(x[1])) == 3 || error("Dim of the first tensor must be 3")
    length(inds(x[end])) == 3 || error("Dim of the last tensor must be 3")
    linkl = Index(1, "Link,l=0")
    linkr = Index(1, "Link,l=$(length(x))")
    x[1] = ITensor(ITensors.data(x[1]), [linkl, inds(x[1])...])
    x[end] = ITensor(ITensors.data(x[end]), [inds(x[end])..., linkr])
    return nothing
end

"""
Decompose the given tensor into as the product of tensors by QR

The externel indices of the results tensors are specified by `inds_list`.
"""
function split_tensor(tensor::ITensor, inds_list::Vector{Vector{Index{T}}}) where {T}
    inds_list = deepcopy(inds_list)
    result = ITensor[]
    for (i, inds) in enumerate(inds_list)
        if i == length(inds_list)
            push!(result, tensor)
        else
            Q, R, _ = qr(tensor, inds)
            push!(result, Q)
            if i < length(inds_list)
                push!(inds_list[i + 1], commonind(Q, R))
            end
            tensor = R
        end
    end
    return result
end

function cleanup_linkinds!(M)
    links_new = [Index(dim(l), "Link,l=$idx") for (idx, l) in enumerate(linkinds(M))]
    links_old = linkinds(M)
    for n in 1:length(M)
        if n < length(M)
            replaceind!(M[n], links_old[n], links_new[n])
        end
        if n > 1
            replaceind!(M[n], links_old[n - 1], links_new[n - 1])
        end
    end
    M
end

"""
To bits
"""
function tobin!(x::Int, xbin::Vector{Int})
    nbit = length(xbin)
    mask = 1 << (nbit - 1)
    for i in 1:nbit
        xbin[i] = (mask & x) >> (nbit - i)
        mask = mask >> 1
    end
end

function tobin(x::Int, R::Int)
    bin = zeros(Int, R)
    tobin!(x, bin)
    return bin
end

# Get bit at pos (>=0). pos=0 is the least significant digit.
_getbit(i, pos) = ((i & (1 << pos)) >> pos)

isascendingorder(x) = issorted(x; lt=isless)
isdecendingorder(x) = issorted(x; lt=Base.isgreater)

isascendingordescending(x) = isascendingorder(x) || isdecendingorder(x)

function kronecker_deltas(sitesin; sitesout=prime.(noprime.(sitesin)))
    N = length(sitesout)
    links = [Index(1, "Link,l=$l") for l in 0:N]
    M = MPO([delta(links[n], links[n + 1], sitesout[n], sitesin[n]) for n in 1:N])
    M[1] *= onehot(links[1] => 1)
    M[end] *= onehot(links[end] => 1)
    return M
end

"""
Match MPS/MPO to the given site indices

MPS:
The resultant MPS do not depends on the missing site indices.

MPO:
For missing site indices, identity operators are inserted.
"""
function matchsiteinds(M::Union{MPS,MPO}, sites)
    N = length(sites)
    sites = noprime.(sites)
    positions = Int[findfirst(sites, s) for s in siteinds(M)]
    if length(M) > 1 && issorted(positions; lt=Base.isgreater)
        return matchsiteinds(MPO([M[n] for n in reverse(1:length(M))]), sites)
    end

    Quantics.isascendingorder(positions) ||
        error("siteinds are not in ascending order!")

    # Add edges
    M_ = deepcopy(M)
    linkl = Index(1, "Link,l=0")
    linkr = Index(1, "Link,l=$N")
    M_[1] = ITensor(ITensors.data(M_[1]), [linkl, inds(M_[1])...])
    M_[end] = ITensor(ITensors.data(M_[end]), [inds(M_[end])..., linkr])

    linkdims_org = [1, dim.(linkinds(M))..., 1]
    linkdims_new = [1, zeros(Int, N - 1)..., 1]
    for n in eachindex(positions)
        p = positions[n]
        linkdims_new[p] = linkdims_org[n]
        linkdims_new[p + 1] = linkdims_org[n + 1]
    end

    # Fill gaps
    while any(linkdims_new .== 0)
        for n in eachindex(linkdims_new)
            if linkdims_new[n] == 0
                if n >= 2 && linkdims_new[n - 1] != 0
                    linkdims_new[n] = linkdims_new[n - 1]
                elseif n < length(linkdims_new) && linkdims_new[n + 1] != 0
                    linkdims_new[n] = linkdims_new[n + 1, 1]
                end
            end
        end
    end

    links = [Index(linkdims_new[l], "Link,l=$(l-1)") for l in eachindex(linkdims_new)]
    if M isa MPO
        tensors = [delta(links[n], links[n + 1]) * delta(sites[n], sites[n]')
                   for n in eachindex(sites)]
    elseif M isa MPS
        tensors = [delta(links[n], links[n + 1]) * ITensor(1, sites[n])
                   for n in eachindex(sites)]
    end

    links_old = [linkl, linkinds(M)..., linkr]
    for n in eachindex(positions)
        p = positions[n]
        tensor = copy(M_[n])
        replaceind!(tensor, links_old[n], links[p])
        replaceind!(tensor, links_old[n + 1], links[p + 1])
        if M isa MPO
            tensors[p] = permute(tensor, [links[p], links[p + 1], sites[p], sites[p]'])
        elseif M isa MPS
            tensors[p] = permute(tensor, [links[p], links[p + 1], sites[p]])
        end
    end

    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)

    return typeof(M)(tensors)
end

asMPO(M::MPO) = M

function asMPO(tensors::Vector{ITensor})
    N = length(tensors)
    M = MPO(N)
    for n in 1:N
        M[n] = tensors[n]
    end
    return M
end

function asMPO(M::MPS)
    return asMPO(ITensors.data(M))
end

function asMPS(M::MPO)
    return MPS([t for t in M])
end

"""
Contract two adjacent tensors in MPO
"""
function combinesites(M::MPO, site1::Index, site2::Index)
    p1 = findsite(M, site1)
    p2 = findsite(M, site2)
    p1 === nothing && error("Not found $site1")
    p2 === nothing && error("Not found $site2")
    abs(p1 - p2) == 1 ||
        error("$site1 and $site2 are found at indices $p1 and $p2. They must be on two adjacent sites.")
    tensors = ITensors.data(M)
    idx = min(p1, p2)
    tensor = tensors[idx] * tensors[idx + 1]
    deleteat!(tensors, idx:(idx + 1))
    insert!(tensors, idx, tensor)
    return MPO(tensors)
end

function combinesites(
        sites::Vector{Vector{Index{IndsT}}},
        site1::AbstractVector{Index{IndsT}},
        site2::AbstractVector{Index{IndsT}}
) where {IndsT}
    length(site1) == length(site2) || error("Length mismatch")
    for (s1, s2) in zip(site1, site2)
        sites = combinesites(sites, s1, s2)
    end
    return sites
end

function combinesites(
        sites::Vector{Vector{Index{IndsT}}}, site1::Index, site2::Index
) where {IndsT}
    sites = deepcopy(sites)
    p1 = findfirst(x -> x[1] == site1, sites)
    p2 = findfirst(x -> x[1] == site2, sites)
    if p1 === nothing || p2 === nothing
        error("Site not found")
    end
    if abs(p1 - p2) != 1
        error("Sites are not adjacent")
    end
    deleteat!(sites, min(p1, p2))
    deleteat!(sites, min(p1, p2))
    insert!(sites, min(p1, p2), [site1, site2])
    return sites
end

function directprod(::Type{T}, sites, indices) where {T}
    length(sites) == length(indices) || error("Length mismatch between sites and indices")
    any(0 .== indices) && error("indices must be 1-based")
    R = length(sites)

    links = [Index(1, "Link,l=$l") for l in 0:R]
    tensors = ITensor[]
    for n in 1:R
        push!(tensors, onehot(links[n] => 1, links[n + 1] => 1, sites[n] => indices[n]))
    end
    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)
    return MPS(tensors)
end

function _find_target_sites(M::MPS; sitessrc=nothing, tag="")
    _find_target_sites(siteinds(M); sitessrc, tag)
end

function _find_target_sites(
        sites::AbstractVector{Index{T}}; sitessrc=nothing, tag="") where {T}
    if tag == "" && sitessrc === nothing
        error("tag or sitesrc must be specified")
    elseif tag != "" && sitessrc !== nothing
        error("tag and sitesrc are exclusive")
    end

    # Set input site indices
    if tag != ""
        sitepos = findallsites_by_tag(sites; tag=tag)
        target_sites = [sites[p] for p in sitepos]
    elseif sitessrc !== nothing
        target_sites = sitessrc
        sitepos = Int[findfirst(x -> x == s, sites) for s in sitessrc]
    end

    return sitepos, target_sites
end

function replace_siteinds_part!(M::MPS, sitesold, sitesnew)
    length(sitesold) == length(sitesnew) ||
        error("Length mismatch between sitesold and sitesnew")

    for i in eachindex(sitesold)
        p = findsite(M, sitesold[i])
        if p === nothing
            error("Not found $(sitesold[i])")
        end
        replaceinds!(M[p], sitesold[i] => sitesnew[i])
    end

    return nothing
end

"""
Connect two MPS's
ITensor objects are deepcopied.
"""
function _directprod(M1::MPS, Mx::MPS...)::MPS
    M2 = Mx[1]
    l = Index(1, "Link")
    tensors1 = [deepcopy(x) for x in M1]
    tensors2 = [deepcopy(x) for x in M2]
    tensors1[end] = ITensor(ITensors.data(last(tensors1)), [inds(last(tensors1))..., l])
    tensors2[1] = ITensor(ITensors.data(first(tensors2)), [l, inds(first(tensors2))...])

    M12 = MPS([tensors1..., tensors2...])
    if length(Mx) == 1
        return M12
    else
        return _directprod(M12, Mx[2:end]...)
    end
end

function rearrange_siteinds(M::AbstractMPS, sites::Vector{Vector{Index{T}}})::MPS where {T}
    sitesold = siteinds(MPO(collect(M)))

    Set(Iterators.flatten(sites)) == Set(Iterators.flatten(sitesold)) ||
        error("siteinds do not match $(sites) != $(sitesold)")

    t = ITensor(1)
    tensors = Vector{ITensor}(undef, length(sites))
    tensors_old = collect(M)
    for (i, site) in enumerate(sites)
        for ind in site
            if ind ∈ inds(t)
                continue
            end
            contract_until = findfirst(x -> ind ∈ Set(collect(x)), inds.(tensors_old))
            contract_until !== nothing || error("ind $ind not found")
            for j in 1:contract_until
                t *= tensors_old[j]
            end
            for _ in 1:contract_until
                popfirst!(tensors_old)
            end
        end

        linds = if i > 1
            vcat(only(commoninds(t, tensors[i - 1])), sites[i])
        else
            sites[i]
        end
        tensors[i], t, _ = qr(t, linds)
    end
    tensors[end] *= t
    cleanup_linkinds!(MPS(tensors))
end

"""
Makes an MPS/MPO diagonal for a specified a site index `s`.
On return, the data will be deep copied and the target core tensor will be diagonalized with an additional site index `s'`.
"""
function makesitediagonal(M::AbstractMPS, site::Index{T})::MPS where {T}
    M_ = deepcopy(MPO(collect(M)))

    target_site::Int = only(findsites(M_, site))
    M_[target_site] = _asdiagonal(M_[target_site], site)

    return MPS(collect(M_))
end

function makesitediagonal(M::AbstractMPS, tag::String)::MPS
    M_ = deepcopy(MPO(collect(M)))
    sites = siteinds(M_)

    target_positions = findallsites_by_tag(siteinds(M_); tag=tag)

    for t in eachindex(target_positions)
        i, j = target_positions[t]
        M_[i] = _asdiagonal(M_[i], sites[i][j])
    end

    return MPS(collect(M_))
end

# FIXME: may be type unstable
function _find_site_allplevs(tensor::ITensor, site::Index; maxplev=10)
    ITensors.plev(site) == 0 || error("Site index must be unprimed.")
    return [ITensors.prime(site, plev)
            for
            plev in 0:maxplev if ITensors.prime(site, plev) ∈ ITensors.inds(tensor)]
end

"""
Extract diagonal components
"""
function extractdiagonal(M::AbstractMPS, tag::String)::MPS
    M_ = deepcopy(MPO(collect(M)))
    sites = siteinds(M_)

    target_positions = findallsites_by_tag(siteinds(M_); tag=tag)

    for t in eachindex(target_positions)
        i, j = target_positions[t]
        M_[i] = _extract_diagonal(M_[i], sites[i][j], sites[i][j]')
    end

    return MPS(collect(M_))
end

function _extract_diagonal(t, site::Index{T}, site2::Index{T}) where {T<:Number}
    dim(site) == dim(site2) || error("Dimension mismatch")
    restinds = uniqueinds(inds(t), site, site2)
    newdata = zeros(eltype(t), dim.(restinds)..., dim(site))
    olddata = Array(t, restinds..., site, site2)
    for i in 1:dim(site)
        newdata[.., i] = olddata[.., i, i]
    end
    return ITensor(newdata, restinds..., site)
end

function _apply(A::MPO, Ψ::MPO; alg::String="fit", cutoff::Real=1e-25, kwargs...)::MPO
    if :algorithm ∈ keys(kwargs)
        error("keyword argument :algorithm is not allowed")
    end
    if alg == "densitymatrix" && cutoff <= 1e-10
        @warn "cutoff is too small for densitymatrix algorithm. Use fit algorithm instead."
    end
    AΨ = replaceprime(
        FastMPOContractions.contract_mpo_mpo(A', asMPO(Ψ); alg, cutoff, kwargs...), 2 => 1)
    MPO(collect(AΨ))
end

function _apply(A::MPO, Ψ::MPS; alg::String="fit", cutoff::Real=1e-25, kwargs...)::MPS
    if :algorithm ∈ keys(kwargs)
        error("keyword argument :algorithm is not allowed")
    end
    if alg == "densitymatrix" && cutoff <= 1e-10
        @warn "cutoff is too small for densitymatrix algorithm. Use fit algorithm instead."
    end
    AΨ = noprime.(FastMPOContractions.contract_mpo_mpo(A, asMPO(Ψ); alg, cutoff, kwargs...))
    MPS(collect(AΨ))
end
