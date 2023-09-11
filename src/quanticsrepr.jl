"""
Quantics index (1-based, ≤2^D) for expanding a D-dimensional space
"""
struct QuanticsInd{D}
    data::Int
    function QuanticsInd{D}(i) where {D}
        1 ≤ i ≤ 2^D || error("Invalid i for QuanticsInd{$D}: $i")
        return new{D}(i)
    end
end

Base.convert(::Type{Int}, x::QuanticsInd{D}) where {D} = x.data
const QubitInd = QuanticsInd{1}
quanticsdim(::QuanticsInd{D}) where {D} = D

Base.convert(::Type{QubitInd}, x::T) where {T<:Integer} = QubitInd(x)

"""
D QubitInd objects are packed into a single QuanticsInd{D} object,
which represent a single length-scale layer of the `D`-dimensional space.
"""
function QuanticsInd{D}(inds::NTuple{D,QubitInd}) where {D}
    i = 0
    c = 2^(D - 1)
    for n in 1:D
        i += c * (convert(Int, inds[D - n + 1]) - 1)
        c = c >> 1
    end
    return QuanticsInd{D}(i + 1)
end

function QuanticsInd(::Val{D}, inds::AbstractVector{<:Integer})::Vector{QuanticsInd{D}} where {D}
    return [QuanticsInd{D}(i) for i in inds]
end

# Deprecated?
#function asqubits(inds::NTuple{D,Int}) where {D}
#return Tuple(QubitInd.(inds))
#end

"""
Convert an integer index (1 <= index <= 2^R) into R qbits
"""
function index_to_qubit(index::Int, R::Int)
    1 <= index <= 2^R || error("index is out of range")

    bits = Vector{Int}(undef, R)
    index_ = index - 1

    for i in 1:R
        bits[R - i + 1] = mod(index_, 2) + 1
        index_ = index_ ÷ 2
    end

    return QubitInd.(bits)
end

"""
Convert a fused quantics index to qubit indices

# 2D case

  - QuanticsInd{2}(1) => (QuanticsInd{1}(1), QuanticsInd{1}(1))
  - QuanticsInd{2}(2) => (QuanticsInd{1}(2), QuanticsInd{1}(1))

Column major order is used to unfuse a quantics index into qubit indices.
"""
function fused_quantics_to_qubit(idx::QuanticsInd{D}) where {D}
    i = convert(Int, idx)
    b = zeros(MVector{D,Int})
    i -= 1
    for n in 1:D
        i, b[n] = divrem(i, 2)
    end
    b .+= 1
    return NTuple{D,QubitInd}(b)
end

"""
Convert fused quantics indices to qubit indices

QuanticsInd{2}.([1, 2, 3, 4]) => QubitInd.([1, 1,  2, 1,  1, 2,  2, 2])
"""
function fused_quantics_to_qubit(indices::AbstractVector{
    QuanticsInd{N}
})::Vector{QubitInd
} where {N
}
    return collect(Iterators.flatten(fused_quantics_to_qubit.(indices)))
end

"""
Convert qubit indices to fused quantics indices
"""
function qubit_to_fused_quantics(::Val{D},
    indices::AbstractVector{QubitInd})::Vector{QuanticsInd{D}
} where {D}
    indices_ = reshape(indices, D, :)
    nquantics = size(indices_, 2)
    return collect(QuanticsInd{D}(Tuple(indices_[:, n])) for n in 1:nquantics)
end

"""
Convert fused quantics indices to integer indices
"""
function fused_quantics_to_index(inds::AbstractVector{QuanticsInd{D}}) where {D}
    R = length(inds) # Number of length-scale layers
    qubits = reshape(fused_quantics_to_qubit(inds), D, R)

    return Tuple(qubit_to_index(qubits[n, :]) for n in 1:D)
end

"""
indices:
(x_1, x_2, ..., x_D), where 1 <= x_i <= 2^R. Reprenents a point in D-dimensional space.

Return:
Fused quantics indices
"""
function index_to_fused_quantics(indices::NTuple{D,Int}, R::Int) where {D}
    qubit_strings = index_to_qubit.(indices, R)
    return qubit_to_fused_quantics(Val(D),
        collect(Iterators.flatten(zip(qubit_strings...))))
end

"""
Convert qubit indices for one-dimensional space into an integer index
"""
function qubit_to_index(inds::AbstractVector{QubitInd})::Int
    R = length(inds)
    c = 2^(R - 1)
    res = 1 # 1-base
    for iq in eachindex(inds)
        res += c * (convert(Int, inds[iq]) - 1)
        c = c >> 1
    end
    return res
end

"""
Convert qubit indices for D-dimensional space into integer indices of size D
"""
function qubit_to_index(::Val{D}, inds::AbstractVector{QubitInd}, R::Int) where {D}
    D * R == length(inds) || error("Length of inds does not match R*D")

    return NTuple{D,Int}(qubit_to_index(inds[i:D:end]) for i in 1:D)
end

"""
Convert integer indices of size D representing a point in D-dimensional space into qubit indices
"""
function index_to_qubit(indices::NTuple{D,Int}, R::Int) where {D}
    return fused_quantics_to_qubit(index_to_fused_quantics(indices, R))
end
