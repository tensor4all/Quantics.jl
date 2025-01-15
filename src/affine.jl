"""
    affine_transform_mpo(y, x, A, b)

Construct and return ITensor matrix product state for the affine transformation
`y = A*x + b` in a (fused) quantics representation:

    y[1,1] ... y[1,M]    y[2,1] ... y[2,M]            y[R,1] ... y[R,M]
     __|__________|__     __|__________|__             __|__________|__
    |                |   |                |           |                |
    |      T[1]      |---|      T[2]      |--- ... ---|      T[R]      |
    |________________|   |________________|           |________________|
       |          |         |          |                 |          |
    x[1,1] ... x[1,N]    x[2,1] ... x[2,N]            x[R,1] ... x[R,N]


Arguments
---------
- `y`: An `R × M` matrix of ITensor indices, where `y[r,m]` corresponds to
  the `r`-th length scale of the `m`-th output variable.
- `x`: An `R × N` matrix of ITensor indices, where `x[r,n]` corresponds to
  the `r`-th length scale of the `n`-th input variable.
- `A`: An `M × N` rational matrix representing the linear transformation.
- `b`: An `M` reational vector representing the translation.
"""
function affine_transform_mpo(
            y::AbstractMatrix{<:Index}, x::AbstractMatrix{<:Index},
            A::AbstractMatrix{<:Union{Integer,Rational}},
            b::AbstractVector{<:Union{Integer,Rational}}
            )::MPO
    R = size(y, 1)
    M, N = size(A)
    size(x) == (R, N) ||
        throw(ArgumentError("insite is not correctly dimensioned"))
    size(y) == (R, M) ||
        throw(ArgumentError("outsite is not correctly dimensioned"))
    size(b) == (M,) ||
        throw(ArgumentError("vector is not correctly dimensioned"))

    # get the tensors so that we know how large the links must be
    tensors = affine_transform_tensors(R, A, b)

    # Create the links
    link = [Index(size(tensors[r], 2), tags="link $r") for r in 1:R-1]

    # Fill the MPO, taking care to not include auxiliary links at the edges
    mpo = MPO(R)
    spin_dims = ntuple(_ -> 2, M + N)
    mpo[1] = ITensor(reshape(tensors[1], size(tensors[1], 2), spin_dims...),
                     (link[1], y[1,:]..., x[1,:]...))
    for r in 2:R-1
        newshape = size(tensors[r])[1:2]..., spin_dims...
        mpo[r] = ITensor(reshape(tensors[r], newshape),
                         (link[r-1], link[r], y[r,:]..., x[r,:]...))
    end
    mpo[R] = ITensor(reshape(tensors[R], size(tensors[R], 1), spin_dims...),
                     (link[R-1], y[R,:]..., x[R,:]...))
    return mpo
end

"""
    affine_transform_tensors(R, A, b)

Compute vector of core tensors (constituent 4-way tensors) for a matrix product
operator corresponding to one of affine transformation `y = A*x + b` with
rational `A` and `b`
"""
function affine_transform_tensors(
            R::Integer, A::AbstractMatrix{<:Union{Integer,Rational}},
            b::AbstractVector{<:Union{Integer,Rational}})
    return affine_transform_tensors(Int(R), _affine_static_args(A, b)...)
end

function affine_transform_tensors(
            R::Int, A::SMatrix{M, N, Int}, b::SVector{M, Int}, s::Int
            ) where {M, N}
    # Checks
    0 <= R <= 8 * sizeof(Int) ||
        throw(DomainError(R, "invalid value of the length R"))
    isodd(s) ||
        throw(DomainError(s, "must be one for now"))

    # We are currently assuming periodic boundary conditions and s being odd.
    # Then there is a multiplicative inverse such that inv_s * s ≡ 1 (mod 2^R)
    # This lets us rewrite: 1/s * (A*x + b) to inv_s*(A*x + b)
    base = 1 << R
    inv_s = modular_inverse(s, base)
    A = inv_s * A
    b = inv_s * b

    # The output tensors are a collection of matrices, but their first two
    # dimensions (links) vary
    tensors = Vector{Array{Bool, 4}}(undef, R)

    # The initial carry is zero
    carry = [zero(SVector{M, Int})]
    for r in R:-1:1
        # Figure out the current bit to add from the shift term and shift
        # it out from the array
        bcurr = @. Bool(b & 1)

        # Get tensor.
        new_carry, data = affine_transform_core(A, bcurr, carry)

        # For the first tensor, we assume periodic boundary conditions, so
        # we sum over all choices off the carry
        if r == 1
            tensors[r] = sum(data, dims=1)
        else
            tensors[r] = data
        end

        # Set carry to the next value
        carry = new_carry
        b = @. b >> 1
    end
    return tensors
end

"""
    core, out_carry = affine_transform_core(A, b, s, in_carry)

Construct core tensor `core` for an affine transform.  The core tensor for an
affine transformation is given by:

    core[d, c, iy, ix] =
        2 * out_carry[d] + s * y[iy] == A * x[ix] + b + in_carry[c]

where `A`, a matrix of integers, and `b`, a vector of bits, which define the
affine transform. `c` and `d` are indices into a set of integer vectors
`in_carry` and `out_carry`, respectively, which encode the incoming and outgoing
carry from the other core tensors. `x[ix] ∈ {0,1}^N` and `y[iy] ∈ {0,1}^M`
are the binary input and output vectors, respectively, of the affine transform.
They are indexed in a "little-endian" fashion.
"""
function affine_transform_core(
            A::SMatrix{M, N, Int}, b::SVector{M, Bool},
            carry::AbstractVector{SVector{M, Int}}
            ) where {M, N}

    # The basic idea here is the following: we compute r = A*x + b + c
    # for all "incoming" carrys d and all possible bit vectors, x ∈ {0,1}^N.
    # Then we decompose r = 2*c + y, where y is again a bit vector, y ∈ {0,1}^M,
    # and c is the "outgoing" carry, which may be negative.  We then store this
    # as something like out[d][c, x, y] = true.
    out = Dict{SVector{M, Int}, Array{Bool, 3}}()
    sizehint!(out, length(carry))
    for (c_index, c) in enumerate(carry)
        for (x_index, x) in enumerate(Iterators.product(ntuple(_ -> 0:1, N)...))
            r = A * SVector{N, Bool}(x) + b + SVector{M, Int}(c)
            y = @. Bool(r & 1)
            d = r .>> 1
            y_index = digits_to_number(y) + 1

            d_mat = get!(out, d) do
                return zeros(Bool, length(carry), 1 << M, 1 << N)
            end
            @inbounds d_mat[c_index, y_index, x_index] = true
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

"""
    affine_transform_matrix(R, A, b; periodic=true)

Compute full transformation matrix for the affine transformation `y = A*x + b`,
where `y` is a `M`-vector and `x` is `N`-vector, and each component is in
`{0, 1, ..., 2^R-1}`. `A` is a rational `M × N` matrix and `b` is a rational
`N`-vector.

Return a boolean sparse `2^(R*M) × 2^(R*N)` matrix `A`, which has true entries
whereever the condition is satisfied. The element indices `A[iy, ix]` are
mapped to `x` and `y` as follows:

    iy = 1 + y[1] + y[2] * 2^R + y[3] * 2^(2R) + ... + y[M] * 2^((M-1)*R)
    ix = 1 + x[1] + x[2] * 2^R + x[3] * 2^(2R) + ... + x[N] * 2^((N-1)*R)

If `periodic` is true, then periodic boundary conditions, `y[i] + 2^R = y[i]`,
are used.
"""
function affine_transform_matrix(
            R::Integer, A::AbstractMatrix{<:Union{Integer,Rational}},
            b::AbstractVector{<:Union{Integer,Rational}}; periodic::Bool=true
            )
    return affine_transform_matrix(Int(R), _affine_static_args(A, b)..., periodic)
end

function affine_transform_matrix(
            R::Int, A::SMatrix{M, N, Int}, b::SVector{M, Int},
            s::Int, periodic::Bool) where {M, N}
    # Checks
    0 <= R ||
        throw(DomainError(R, "invalid value of the length R"))
    isodd(s) ||
        throw(DomainError(s, "right now we only support odd s"))

    mask = ~(~0 << R)
    inv_s = modular_inverse(s, R)
    y_index = Int[]
    x_index = Int[]

    for (ix, x) in enumerate(Iterators.product(ntuple(_ -> 0:mask, N)...))
        v = A * SVector{N, Int}(x) + b
        if periodic
            v *= inv_s
            y = v .& mask
        else
            iszero(v .% s) || continue
            y = v .÷ s
        end
        iy = digits_to_number(y, R) + 1
        push!(y_index, iy)
        push!(x_index, ix)
    end
    values = ones(Bool, size(x_index))
    return sparse(y_index, x_index, values, 1 << (R*M), 1 << (R*N))
end

function affine_mpo_to_matrix(
            outsite::AbstractMatrix{<:Index}, insite::AbstractMatrix{<:Index},
            mpo::MPO)
    prev_warn_order = ITensors.disable_warn_order()
    try
        mpo_contr = reduce(*, mpo)

        # Given some variables (x, y), we have to bring the indices in the
        # order (xR, ..., x1, yR, ..., y1) in order to have y = (y1 ... yR)_2
        # once we reshape a column-major array and match the order of the
        # variables in the full matrix.
        out_indices = vec(reverse(outsite, dims=1))
        in_indices = vec(reverse(insite, dims=1))
        tensor = Array(mpo_contr, out_indices..., in_indices...)
        matrix = reshape(tensor,
                         1 << length(out_indices), 1 << length(in_indices))
        return matrix
    finally
        ITensors.set_warn_order(prev_warn_order)
    end
end

function _affine_static_args(A::AbstractMatrix{<:Union{Integer,Rational}},
                             b::AbstractVector{<:Union{Integer,Rational}})
    M, N = size(A)
    size(b, 1) == M ||
        throw(ArgumentError("A and b have incompatible size"))

    # Factor out common denominator and pass
    denom = lcm(mapreduce(denominator, lcm, A, init=1),
                mapreduce(denominator, lcm, b, init=1))
    Ai = @. Int(denom * A)
    bi = @. Int(denom * b)

    # Construct static matrix
    return SMatrix{M, N, Int}(Ai), SVector{M, Int}(bi), denom
end

"""
    Ainv, binv = active_to_passive(A::AbstractMatrix, b::AbstractVector)

Change active affine transformation `y = A*x + b` to the passive (inverse) one
`x = Ainv*y + binv`.

Note that these transformations are not strict inverses of each other once you
put them on a discrete grid: in particular, Ainv may have rational coefficients
even though A is purely integer. In this case, the inverse transformation only
maps some of the points.
"""
function active_to_passive(A::AbstractMatrix{<:Union{Rational,Integer}},
                           b::AbstractVector{<:Union{Rational,Integer}})
    return active_to_passive(Rational.(A), Rational.(b))
end

function active_to_passive(
            A::AbstractMatrix{<:Rational}, b::AbstractVector{<:Rational})
    m, n = size(A)
    T = [A b; zero(b)' 1]

    # XXX: we do not support pseudo-inverses (LinearAlgebbra cannot do
    #      this yet over the Rationals).
    Tinv = inv(T)
    Ainv = Tinv[1:m, 1:n]
    binv = Tinv[1:m, n+1]
    return Ainv, binv
end

"""
    digits_to_number(v::AbstractVector{Bool})
    digits_to_number(v::AbstractVector{<:Integer}, bits::Integer)

Converts a vector of digits, starting with the least significant digit, to
a number.  If the digits are boolean, then they are interpreted in base-2,
otherwise, in base `2^bits`.
"""
digits_to_number(v::AbstractVector{Bool}) = _digits_to_number(Tuple(v))
digits_to_number(v::AbstractVector{<:Integer}, bits::Integer) =
    _digits_to_number(Int.(Tuple(v)), Int(bits))

@inline _digits_to_number(v::Tuple{}) = 0
@inline _digits_to_number(v::Tuple{Bool, Vararg{Bool}}) =
    _digits_to_number(v[2:end]) << 1 | v[1]
@inline _digits_to_number(v::Tuple{}, bits::Int) = 0
@inline function _digits_to_number(v::Tuple{Int, Vararg{Int}}, bits::Int)
    mask = (~0) << bits
    iszero(v[1] & mask) ||
        throw(DomainError(v[1], "invalid digit in base $(1 << bits)"))
    return _digits_to_number(v[2:end], bits) << bits | v[1]
end

"""
    modular_inverse(b, m)

Compute the multiplicative inverse of an integer `b` modulo `2^m`, i.e., find
and return an integer `x` such that:

    x * b ≡ 1  (mod 2^m)
"""
function modular_inverse(b::Integer, m::Integer)
    0 <= m <= 8 * sizeof(b) ||
        throw(DomainError(m, "invalid number of bits"))
    isodd(b) ||
        throw(DomainError(b, "only odd numbers have inverses mod power of 2"))

    # Use Dusse and Kaliski's algorithm, as reproduced in
    # Arazi and Qi, IEEE Trans. Comput. 57, 10 (2008), Algorithm 1
    mask = one(b)
    y = one(b)
    for _ in 2:m
        mask <<= 1
        if (b * y) & mask != 0
            y |= mask
        end
    end
    return y
end
