using StaticArrays
using ITensors

function affine_transform_mpo(
            outsite::AbstractMatrix{<:Index}, insite::AbstractMatrix{<:Index},
            A::AbstractMatrix{<:Integer}, b::AbstractVector{<:Integer}
            )
    R = size(outsite, 1)
    M, N = size(A)
    size(insite) == (R, N) ||
        throw(ArgumentError("insite is not correctly dimensioned"))
    size(b) == (M,) ||
        throw(ArgumentError("vector is not correctly dimensioned"))

    # get the tensors so that we know how large the links must be
    tensors = affine_transform_tensors(
                    R, SMatrix{M, N, Int}(A), SVector{M, Int}(b))

    # Create the links
    link = [Index(size(tensors[r], 1), tags="link $r") for r in 1:R-1]

    # Fill the MPO, taking care to not include auxiliary links at the edges
    mpo = MPO(R)
    spin_dims = ntuple(_ -> 2, M + N)
    mpo[1] = ITensor(reshape(tensors[1], size(tensors[1])[1], spin_dims...),
                     (link[1], outsite[1,:]..., insite[1,:]...))
    for r in 2:R-1
        newshape = size(tensors[r])[1:2]..., spin_dims...
        mpo[r] = ITensor(reshape(tensors[r], newshape),
                         (link[r], link[r-1], outsite[r,:]..., insite[r,:]...))
    end
    println(size(tensors[R]))
    mpo[R] = ITensor(reshape(tensors[R], size(tensors[R])[2], spin_dims...),
                     (link[R-1], outsite[R,:]..., insite[R,:]...))
    return mpo
end

"""
    affine_transform_tensors(R, A, b)

Compute vector of core tensors (constituent 4-way tensors) for a matrix product
operator corresponding to the affine transformation `y = A*x + b`.
"""
function affine_transform_tensors(
            R::Integer, A::AbstractMatrix{<:Integer},
            b::AbstractVector{<:Integer})
    M, N = size(A)
    return affine_transform_tensors(
            Int(R), SMatrix{M, N, Int}(A), SVector{M, Int}(b))
end

function affine_transform_tensors(
            R::Int, A::SMatrix{M, N, Int}, b::SVector{M, Int}
            ) where {M, N}
    # Checks
    0 <= R ||
        throw(DomainError(R, "invalid value of the length R"))

    # The output tensors are a collection of matrices, but their first two
    # dimensions (links) vary
    tensors = Vector{Array{Bool, 4}}(undef, R)

    # The initial carry is zero
    carry = [zero(SVector{M, Int})]
    for r in 1:R
        # Figure out the current bit to add from the shift term and shift
        # it out from the array
        bcurr = @. Bool(b & 1)

        # Get tensor. For the last tensor, we discard the carry.
        out_tf = (r == R) ? zero : identity
        new_carry, data = affine_transform_core(A, bcurr, carry,
                                                transform_out_carry=out_tf)
        tensors[r] = data

        # Set carry to the next value
        carry = new_carry
        b = @. b >> 1
    end
    return tensors
end

"""
    core, out_carry = affine_transform_core(A, b, in_carry;
                                            transform_out_carry=identity)

Construct core tensor `core` for an affine transform.  The core tensor for an
affine transformation is given by:

    core[d, c, iy, ix] =
        2 * out_carry[d] + y[iy] == A * x[ix] + b + in_carry[c]

where `A`, a matrix of integers, and `b`, a vector of bits, which define the
affine transform. `c` and `d` are indices into a set of integer vectors
`in_carry` and `out_carry`, respectively, which encode the incoming and outgoing
carry from the other core tensors. `x[ix] ∈ {0,1}^N` and `y[iy] ∈ {0,1}^M`
are the binary input and output vectors, respectively, of the affine transform.
They are indexed in a "little-endian" fashion.
"""
function affine_transform_core(
            A::SMatrix{M, N, Int}, b::SVector{M, Bool},
            carry::AbstractVector{SVector{M, Int}};
            transform_out_carry::Function=identity
            ) where {M, N}
    # The basic idea here is the following: we compute r = A*x + b + c for all
    # "incoming" carrys d and all possible bit vectors, x ∈ {0,1}^N.  Then we
    # decompose r = 2*c + y, where y is again a bit vector, y ∈ {0,1}^M, and
    # c is the "outgoing" carry, which may be negative.  We then store this
    # as something like out[d][c, x, y] = true.
    out = Dict{SVector{M, Int}, Array{Bool, 3}}()
    sizehint!(out, length(carry))
    for (c_index, c) in enumerate(carry)
        for (x_index, x) in enumerate(Iterators.product(ntuple(_ -> 0:1, N)...))
            r = A * SVector{N, Bool}(x) + b + SVector{N, Int}(c)
            y = @. Bool(r & 1)
            d::SVector{M, Int} = transform_out_carry(r .>> 1)
            y_index = _spins_to_number(y) + 1

            d_mat = get!(out, d) do
                return zeros(Bool, length(carry), 1 << M, 1 << N)
            end
            @inbounds d_mat[c_index, x_index, y_index] = true
        end
    end

    # We translate the dictionary into a vector of carrys (which we can then
    # pass into the next iteration) and a 4-way tensor of output values.
    carry_out = Vector{SVector{M, Int}}(undef, length(out))
    value_out = Array{Bool, 4}(undef, length(out), length(carry), 1<<M, 1<<N)
    for (p_index, p) in enumerate(pairs(out))
        carry_out[p_index] = p.first
        value_out[p_index, :, :, :] .= p.second
    end
    return carry_out, value_out
end

_spins_to_number(v::SVector{<:Any, Bool}) = _spins_to_number(Tuple(v))
_spins_to_number(v::Tuple{Bool, Vararg{Bool}}) =
    _spins_to_number(v[2:end]) << 1 | v[1]
_spins_to_number(v::Tuple{}) = 0
