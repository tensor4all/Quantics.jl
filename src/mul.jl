abstract type AbstractMultiplier end

#===
Matrix multiplication
===#
struct MatrixMultiplier{T} <: AbstractMultiplier where {T}
    sites_row::Vector{Index{T}}
    sites_shared::Vector{Index{T}}
    sites_col::Vector{Index{T}}

    function MatrixMultiplier(sites_row::Vector{Index{T}},
            sites_shared::Vector{Index{T}},
            sites_col::Vector{Index{T}}) where {T}
        new{T}(sites_row, sites_shared, sites_col)
    end
end

function MatrixMultiplier(site_row::Index{T},
        site_shared::Index{T},
        site_col::Index{T}) where {T}
    return MatrixMultiplier([site_row], [site_shared], [site_col])
end

function preprocess(mul::MatrixMultiplier{T}, M1::MPO, M2::MPO) where {T}
    for (site_row, site_shared, site_col) in zip(mul.sites_row, mul.sites_shared,
        mul.sites_col)
        M1, M2 = combinesites(M1, site_row, site_shared),
        combinesites(M2, site_col, site_shared)
    end
    return M1, M2
end

function postprocess(mul::MatrixMultiplier{T}, M::MPO)::MPO where {T}
    tensors = ITensors.data(M)
    for (site_row, site_col) in zip(mul.sites_row, mul.sites_col)
        p = findfirst(hasind(site_row), tensors)
        hasind(tensors[p], site_col) ||
            error("$site_row and $site_col are not on the same site")

        indsl = [site_row]
        if p > 1
            push!(indsl, linkind(M, p - 1))
        end

        indsr = [site_col]
        if p < length(M)
            push!(indsr, linkind(M, p))
        end

        Ml, Mr = split_tensor(tensors[p], [indsl, indsr])

        deleteat!(tensors, p)
        insert!(tensors, p, Ml)
        insert!(tensors, p + 1, Mr)
    end

    return MPO(tensors)
end

#===
Elementwise multiplication
===#
struct ElementwiseMultiplier{T} <: AbstractMultiplier where {T}
    sites::Vector{Index{T}}
    function ElementwiseMultiplier(sites::Vector{Index{T}}) where {T}
        new{T}(sites)
    end
end

"""
Convert an MPS tensor to an MPO tensor with a diagonal structure
"""
function _asdiagonal(t, site::Index{T}; baseplev=0)::ITensor where {T<:Number}
    ITensors.hasinds(t, site') && error("Found $(site')")
    links = ITensors.uniqueinds(ITensors.inds(t), site)
    rawdata = Array(t, links..., site)
    tensor = zeros(eltype(t), size(rawdata)..., ITensors.dim(site))
    for i in 1:ITensors.dim(site)
        tensor[.., i, i] = rawdata[.., i]
    end
    return ITensor(
        tensor, links..., ITensors.prime(site, baseplev + 1), ITensors.prime(
            site, baseplev)
    )
end

function _todense(t, site::Index{T}) where {T<:Number}
    links = uniqueinds(inds(t), site, site'')
    newdata = zeros(eltype(t), dim.(links)..., dim(site))
    if length(links) == 2
        olddata = Array(t, links..., site, site'')
        for i in 1:dim(site)
            newdata[:, :, i] = olddata[:, :, i, i]
        end
    elseif length(links) == 1
        olddata = Array(t, links..., site, site'')
        for i in 1:dim(site)
            newdata[:, i] = olddata[:, i, i]
        end
    else
        error("Too many links found: $links")
    end
    return ITensor(newdata, links..., site)
end

function preprocess(mul::ElementwiseMultiplier{T}, M1::MPO, M2::MPO) where {T}
    tensors1 = ITensors.data(M1)
    tensors2 = ITensors.data(M2)
    for s in mul.sites
        p = findfirst(hasind(s), tensors1)
        hasinds(tensors2[p], s) || error("ITensor of M2 at $p does not have $s")
        #tensors1[p] = replaceprime(_asdiagonal(tensors1[p], s), 0 => 1, 1 => 2)
        tensors1[p] = _asdiagonal(tensors1[p], s)
        replaceind!(tensors1[p], s' => s'')
        replaceind!(tensors1[p], s => s')
        tensors2[p] = _asdiagonal(tensors2[p], s)
    end
    return MPO(tensors1), MPO(tensors2)
end

function postprocess(mul::ElementwiseMultiplier{T}, M::MPO)::MPO where {T}
    tensors = ITensors.data(M)
    for s in mul.sites
        p = findfirst(hasind(s), tensors)
        tensors[p] = _todense(tensors[p], s)
    end
    return MPO(tensors)
end

"""
By default, elementwise multiplication will be performed.
"""
function automul(M1::MPS, M2::MPS; tag_row::String="", tag_shared::String="",
        tag_col::String="", alg="naive", cutoff=1e-30, kwargs...)
    if in(:maxbonddim, keys(kwargs))
        error("Illegal keyward parameter: maxbonddim. Use maxdim instead!")
    end

    sites_row = findallsiteinds_by_tag(siteinds(M1); tag=tag_row)
    sites_shared = findallsiteinds_by_tag(siteinds(M1); tag=tag_shared)
    sites_col = findallsiteinds_by_tag(siteinds(M2); tag=tag_col)
    sites_matmul = Set(Iterators.flatten([sites_row, sites_shared, sites_col]))

    if sites_shared != findallsiteinds_by_tag(siteinds(M2); tag=tag_shared)
        error("Invalid shared sites for MatrixMultiplier")
    end

    matmul = MatrixMultiplier(sites_row, sites_shared, sites_col)
    ewmul = ElementwiseMultiplier([s for s in siteinds(M1) if s ∉ sites_matmul])

    M1_ = Quantics.asMPO(M1)
    M2_ = Quantics.asMPO(M2)
    M1_, M2_ = preprocess(matmul, M1_, M2_)
    M1_, M2_ = preprocess(ewmul, M1_, M2_)

    M = FastMPOContractions.contract_mpo_mpo(M1_, M2_; alg=alg, cutoff=cutoff, kwargs...)

    M = Quantics.postprocess(matmul, M)
    M = Quantics.postprocess(ewmul, M)

    if in(:maxdim, keys(kwargs))
        truncate!(M; maxdim=kwargs[:maxdim], cutoff=cutoff)
    end

    return asMPS(M)
end
