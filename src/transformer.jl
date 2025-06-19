function _single_tensor_flip()
    cval = [-1, 0]
    # (cin, cout, s', s)
    single_tensor = zeros(Float64, 2, 2, 2, 2)
    for icin in 1:2
        for a in 1:2
            out = -(a - 1) + cval[icin]
            icout = out < 0 ? 1 : 2
            b = mod(out, 2) + 1
            single_tensor[icin, icout, a, b] = 1
        end
    end
    return single_tensor
end

"""
This function returns an MPO, M, representing the transformation
f(x) = g(-x)
where f(x) = M * g(x) for x = 0, 1, ..., 2^R-1.
"""
function flipop_to_negativedomain(sites::Vector{Index{T}}; rev_carrydirec=false,
        bc::Int=1)::MPO where {T}
    return flipop(sites; rev_carrydirec=rev_carrydirec, bc=bc) * bc
end

"""
This function returns an MPO, M, representing the transformation
f(x) = g(2^R-x)
where f(x) = M * g(x) for x = 0, 1, ..., 2^R-1.

`sites`: the sites of the output MPS
"""
function flipop(sites::Vector{Index{T}}; rev_carrydirec=false, bc::Int=1)::MPO where {T}
    if rev_carrydirec
        M = flipop(reverse(sites); rev_carrydirec=false, bc=bc)
        return MPO([M[n] for n in reverse(1:length(M))])
    end

    N = length(sites)
    abs(bc) == 1 || error("bc must be either 1, -1")
    N > 1 || error("MPO with one tensor is not supported")

    t = _single_tensor_flip()
    M = MPO(N)
    links = [Index(2, "Link,l=$l") for l in 1:(N + 1)]
    for n in 1:N
        M[n] = ITensor(t, (links[n], links[n + 1], sites[n]', sites[n]))
    end

    M[1] *= onehot(links[1] => 2)

    bc_tensor = ITensor([1.0, bc], links[end])
    M[N] = M[N] * bc_tensor

    cleanup_linkinds!(M)

    return M
end

@doc """
f(x) = g(N - x) = M * g(x) for x = 0, 1, ..., N-1,
where x = 0, 1, ..., N-1 and N = 2^R.

Note that x = 0, 1, 2, ..., N-1 are mapped to x = 0, N-1, N-2, ..., 1 mod N.
"""
function reverseaxis(M::MPS; tag="x", bc::Int=1, kwargs...)
    bc ∈ [1, -1] || error("bc must be either 1 or -1")
    return _apply(reverseaxismpo(siteinds(M); tag=tag, bc=bc), M; kwargs...)
end

function reverseaxismpo(sites::AbstractVector{Index{T}}; tag="x", bc::Int=1)::MPO where {T}
    bc ∈ [1, -1] || error("bc must be either 1 or -1")
    targetsites = findallsiteinds_by_tag(sites; tag=tag)
    pos = findallsites_by_tag(sites; tag=tag)
    !isascendingordescending(pos) && error("siteinds for tag $(tag) must be sorted.")
    rev_carrydirec = isascendingorder(pos)
    siteinds_MPO = rev_carrydirec ? targetsites : reverse(targetsites)
    transformer_tag = flipop(siteinds_MPO; rev_carrydirec=rev_carrydirec, bc=bc)
    return matchsiteinds(transformer_tag, sites)
end

"""
f(x) = g(x + shift) for x = 0, 1, ..., 2^R-1 and 0 <= shift < 2^R.
"""
function shiftaxis(M::MPS, shift::Int; tag="x", bc::Int=1, kwargs...)
    bc ∈ [1, -1] || error("bc must be either 1 or -1")
    return _apply(shiftaxismpo(siteinds(M), shift; tag=tag, bc=bc), M; kwargs...)
end

"""
f(x) = g(x + shift) for x = 0, 1, ..., 2^R-1 and 0 <= shift < 2^R.
"""
function shiftaxismpo(
        sites::AbstractVector{Index{T}}, shift::Int; tag="x", bc::Int=1)::MPO where {T}
    bc ∈ [1, -1] || error("bc must be either 1 or -1")
    targetsites = findallsiteinds_by_tag(sites; tag=tag) # From left to right: x=1, 2, ...
    pos = findallsites_by_tag(sites; tag=tag)
    !isascendingordescending(pos) && error("siteinds for tag $(tag) must be sorted.")
    rev_carrydirec = isascendingorder(pos)

    R = length(targetsites)
    nbc, shift_mod = divrem(shift, 2^R, RoundDown)

    if rev_carrydirec
        transformer = _shift_mpo(targetsites, shift_mod; bc=bc)
    else
        transformer = _shift_mpo(targetsites, shift_mod; bc=bc)
        transformer = MPO([transformer[n] for n in reverse(1:length(transformer))])
    end
    transformer = matchsiteinds(transformer, sites)
    transformer *= bc^nbc

    return transformer
end

"""
Multiply by exp(i θ x), where x = (x_1, ..., x_R)_2.
"""
function phase_rotation(M::MPS, θ::Real; targetsites=nothing, tag="", kwargs...)::MPS
    transformer = phase_rotation_mpo(siteinds(M), θ; targetsites=targetsites, tag=tag)
    _apply(transformer, M; kwargs...)
end

"""
Create an MPO for multiplication by `exp(i θ x)`, where `x = (x_1, ..., x_R)_2`.

`sites`: site indices for `x_1`, `x_2`, ..., `x_R`.
"""
function phase_rotation_mpo(sites::AbstractVector{Index{T}}, θ::Real;
        targetsites=nothing, tag="")::MPO where {T}
    _, target_sites = _find_target_sites(sites; sitessrc=targetsites, tag=tag)
    transformer = _phase_rotation_mpo(target_sites, θ)
    return matchsiteinds(transformer, sites)
end

function _phase_rotation_mpo(sites::AbstractVector{Index{T}}, θ::Real)::MPO where {T}
    R = length(sites)
    tensors = [ITensor(true) for _ in 1:R]
    θ_mod = mod(θ, 2π)
    p_back = precision(BigFloat)
    setprecision(BigFloat, 256)
    for n in 1:R
        tmp = Float64(mod(θ_mod * BigFloat(2)^BigFloat(R - n), 2 * π))
        tensors[n] = ITensors.SiteTypes.op("Phase", sites[n]; ϕ=tmp)
    end
    links = [Index(1, "Link,l=$l") for l in 1:(R - 1)]
    tensors[1] = ITensor(
        Array(tensors[1], sites[1]', sites[1]), sites[1], sites[1]', links[1])
    for l in 2:(R - 1)
        tensors[l] = ITensor(Array(tensors[l], sites[l]', sites[l]),
            links[l - 1], sites[l], sites[l]', links[l])
    end
    tensors[end] = ITensor(
        Array(tensors[end], sites[end]', sites[end]), links[end], sites[end], sites[end]')

    setprecision(BigFloat, p_back)
    return MPO(tensors)
end

function _upper_lower_triangle(upper_or_lower::Symbol)::Array{Float64,4}
    upper_or_lower ∈ [:upper, :lower] || error("Invalid upper_or_lower $(upper_or_lower)")
    T = Float64
    t = zeros(T, 2, 2, 2, 2) # left link, right link, site', site

    t[1, 1, 1, 1] = one(T)
    t[1, 1, 2, 2] = one(T)

    if upper_or_lower == :upper
        t[1, 2, 1, 2] = one(T)
        t[1, 2, 2, 1] = zero(T)
    else
        t[1, 2, 1, 2] = zero(T)
        t[1, 2, 2, 1] = one(T)
    end

    # If a comparison is made at a higher bit, we respect it.
    t[2, 2, :, :] .= one(T)

    return t
end

"""
Create QTT for a upper/lower triangle matrix filled with one except the diagonal line
"""
function upper_lower_triangle_matrix(sites::Vector{Index{T}}, value::S;
        upper_or_lower::Symbol=:upper)::MPO where {T,S}
    upper_or_lower ∈ [:upper, :lower] || error("Invalid upper_or_lower $(upper_or_lower)")
    N = length(sites)

    t = _upper_lower_triangle(upper_or_lower)

    M = MPO(N)
    links = [Index(2, "Link,l=$l") for l in 1:(N + 1)]
    for n in 1:N
        M[n] = ITensor(t, (links[n], links[n + 1], sites[n]', sites[n]))
    end

    M[1] *= onehot(links[1] => 1)
    M[N] *= ITensor(S[0, value], links[N + 1])

    return M
end

"""
Create MPO for cumulative sum in QTT

includeown = False
y_i = sum_{j=1}^{i-1} x_j
"""
function cumsum(sites::Vector{Index}; includeown::Bool=false)
    includeown == False || error("includeown = True has not been implmented yet")
    return upper_triangle_matrix(sites, 1.0)
end

"""
Add new site indices to an MPS
"""
#==
function asdiagonal(M::MPS, newsites; which_new="right", targetsites=nothing, tag="")
    which_new ∈ ["left", "right"] || error("Invalid which_new: left or right")
    sitepos, target_sites = Quantics._find_target_sites(M; sitessrc=targetsites, tag=tag)
    length(sitepos) == length(newsites) ||
        error("Length mismatch: $(newsites) vs $(target_sites)")
    M_ = Quantics._addedges(M)
    links = linkinds(M_)

    tensors = ITensor[]
    for p in 1:length(M)
        if !(p ∈ sitepos)
            push!(tensors, copy(M_[p]))
            continue
        end
        i = findfirst(x -> x == p, sitepos)
        s = target_sites[i]
        s1 = sim(s)
        ll, lr = links[p], links[p + 1]
        t = replaceind(M_[p], s => s1)
        if which_new == "right"
            tl, tr = factorize(delta(s1, s, newsites[i]) * t, ll, s)
        else
            tl, tr = factorize(delta(s1, s, newsites[i]) * t, ll, newsites[i])
        end
        push!(tensors, tl)
        push!(tensors, tr)
    end

    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)

    M_result = MPS(tensors)
    Quantics.cleanup_linkinds!(M_result)
    return M_result
end
==#
