using .TensorCrossInterpolation
import .TensorCrossInterpolation: TensorCI, CachedFunction, TensorCI2
import .TensorCrossInterpolation as TCI

function TCItoMPS(tci::Union{TensorCI{T},TensorCI2{T}}, sites=nothing) where {T}
    tensors = TCI.tensortrain(tci)
    ranks = TCI.rank(tci)
    N = length(tensors)
    localdims = [size(t, 2) for t in tensors]

    if sites === nothing
        sites = [Index(localdims[n], "n=$n") for n in 1:N]
    else
        all(localdims .== dim.(sites)) ||
            error("ranks are not consistent with dimension of sites")
    end

    linkdims = [[size(t, 1) for t in tensors]..., 1]
    links = [Index(linkdims[l + 1], "link,l=$l") for l in 0:N]

    tensors_ = [ITensor(deepcopy(tensors[n]), links[n], sites[n], links[n + 1])
                for n in 1:N]
    tensors_[1] *= onehot(links[1] => 1)
    tensors_[end] *= onehot(links[end] => 1)

    return MPS(tensors_)
end

abstract type AbstractAdaptiveQTTNode end

struct AdaptiveQTTLeaf{T<:Number} <: AbstractAdaptiveQTTNode
    tci::TensorCI{T}
    prefix::Vector{Int}
end

function Base.show(io::IO, obj::AdaptiveQTTLeaf{T}) where {T}
    prefix = convert.(Int, obj.prefix)
    println(io,
            "  "^length(prefix) * 
            "Leaf $(prefix): rank=$(maximum(TCI.linkdims(obj.tci)))")
end

struct AdaptiveQTTInternalNode{T<:Number} <: AbstractAdaptiveQTTNode
    children::Dict{Int,AbstractAdaptiveQTTNode}
    prefix::Vector{Int}

    function AdaptiveQTTInternalNode{T}(children::Dict{Int,AbstractAdaptiveQTTNode}, prefix::Vector{Int}) where {T}
        return new{T}(children, prefix)
    end
end

"""
prefix is the common prefix of all children
"""
function AdaptiveQTTInternalNode{T}(children::Vector{AbstractAdaptiveQTTNode},
                                    prefix::Vector{Int}) where {T}
    d = Dict{Int,AbstractAdaptiveQTTNode}()
    for child in children
        d[child.prefix[end]] = child
    end
    return AdaptiveQTTInternalNode{T}(d, prefix)
end

function Base.show(io::IO, obj::AdaptiveQTTInternalNode{T}) where {T}
    println(io,
        "  "^length(obj.prefix) * 
        "InternalNode $(obj.prefix) with $(length(obj.children)) children")
    for (k, v) in obj.children
        Base.show(io, v)
    end
end


"""
Evaluate the tree at given idx
"""
function evaluate(obj::AdaptiveQTTInternalNode{T}, idx::AbstractVector{Int})::T where {T}
    child_key = idx[length(obj.prefix) + 1]
    return evaluate(obj.children[child_key], idx)
end

function evaluate(obj::AdaptiveQTTLeaf{T}, idx::AbstractVector{Int})::T where {T}
    return TCI.evaluate(obj.tci, idx[length(obj.prefix) + 1:end])
end

"""
Convert a dictionary of patches to a tree
"""
function _to_tree(patches::Dict{Vector{Int},TensorCI{T}}; nprefix=0)::AbstractAdaptiveQTTNode where {T}
    length(unique(k[1:nprefix] for (k, v) in patches)) == 1 || error("Inconsistent prefixes")

    common_prefix = first(patches)[1][1:nprefix]

    # Return a leaf
    if nprefix == length(first(patches)[1])
        return AdaptiveQTTLeaf{T}(first(patches)[2], common_prefix)
    end

    subgroups = Dict{Int, Dict{Vector{Int},TensorCI{T}}}()
    
    # Look at the first index after nprefix skips
    # and group the patches by that index
    for (k, v) in patches
        idx = k[nprefix + 1]
        if idx in keys(subgroups)
            subgroups[idx][k] = v
        else
            subgroups[idx] = Dict{Vector{Int},TensorCI{T}}(k=>v)
        end
    end

    # Recursively construct the tree
    children = AbstractAdaptiveQTTNode[]
    for (_, grp) in subgroups
        push!(children, _to_tree(grp; nprefix=nprefix+1))
    end

    return AdaptiveQTTInternalNode{T}(children, common_prefix)
end


"""
Construct QTTs using adaptive partitioning of the domain.

TODO
* Use crossinterpolate2
* Allow arbitrary order of partitioning
* Parallelization
"""
function construct_adaptiveqtt2(::Type{T}, f::Function, localdims::AbstractVector{Int}; maxiter=100, firstpivot=ones(Int, length(localdims)), kwargs...)::AdaptiveQTTInternalNode{T} where T
    R = length(localdims)
    leaves = Dict{Vector{Int},TensorCI{T}}()

    # Add root node
    firstpivot = TCI.optfirstpivot(f, localdims, firstpivot)
    tci, ranks, errors = crossinterpolate(T, f, localdims,
                       firstpivot;
                       maxiter=maxiter,
                       kwargs...)
    leaves[[]] = tci

    while true
        done = true
        for (prefix, tci) in leaves
            if maximum(TCI.linkdims(tci)) >= maxiter ÷ 2
                done = false
                delete!(leaves, prefix)
                for ic in 1:localdims[length(prefix)+1]
                    prefix_ = vcat(prefix, ic)
                    localdims_ = localdims[length(prefix_)+1:end]
                    f_ = x -> f(vcat(prefix_, x))
                    firstpivot_ = ones(Int, R - length(prefix_))
                    firstpivot_ = TCI.optfirstpivot(f_, localdims_, firstpivot_)
                    t_, ranks_, errors_ = crossinterpolate(T,
                                       f_,
                                       localdims_,
                                       firstpivot_;
                                       maxiter=maxiter,
                                       kwargs...)
                    leaves[prefix_] = t_
                end
            end
        end
        if done
            break
        end
    end

    return _to_tree(leaves)
end