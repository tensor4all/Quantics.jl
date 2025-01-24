"""
    AbstractBoundaryConditions

Boundary conditions for the QTT to use. Use `OpenBoundaryCondtions`` for open boundaries and `PeriodicBoundaryConditions` for periodic ones.
"""
abstract type AbstractBoundaryConditions end

struct PeriodicBoundaryConditions <: AbstractBoundaryConditions end

@inline function equiv(x::SVector, y::SVector, R::Int, ::PeriodicBoundaryConditions)
    mask = ~(~0 << R)
    return iszero((x - y) .& mask)
end

carry_weight(c::SVector, ::PeriodicBoundaryConditions) = true

struct OpenBoundaryConditions <: AbstractBoundaryConditions end

equiv(x::SVector, y::SVector, R::Int, ::OpenBoundaryConditions) = x == y

carry_weight(c::SVector, ::OpenBoundaryConditions) = iszero(c)

"""
    affine_transform_mpo(y, x, A, b, [boundary])

Construct and return ITensor matrix product state for the affine transformation
`y = A*x + b` in a (fused) quantics representation:

    y[1,1] ... y[1,M]    y[2,1] ... y[2,M]            y[R,1] ... y[R,M]
     __|__________|__     __|__________|__             __|__________|__
    |                |   |                |           |                |
    |      T[1]      |---|      T[2]      |--- ... ---|      T[R]      |
    |________________|   |________________|           |________________|
       |          |         |          |                 |          |
    x[1,1] ... x[1,N]    x[2,1] ... x[2,N]            x[R,1] ... x[R,N]

## Arguments

  - `y`: An `R × M` matrix of ITensor indices, where `y[r,m]` corresponds to
    the `r`-th length scale of the `m`-th output variable.
  - `x`: An `R × N` matrix of ITensor indices, where `x[r,n]` corresponds to
    the `r`-th length scale of the `n`-th input variable.
  - `A`: An `M × N` rational matrix representing the linear transformation.
  - `b`: An `M` reational vector representing the translation.
  - `boundary`: boundary conditions (defaults to `PeriodicBoundaryConditions()`)
"""
function affine_transform_mpo(
        y::AbstractMatrix{<:Index}, x::AbstractMatrix{<:Index},
        A::AbstractMatrix{<:Union{Integer,Rational}},
        b::AbstractVector{<:Union{Integer,Rational}},
        boundary::AbstractBoundaryConditions=PeriodicBoundaryConditions()
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
    tensors = affine_transform_tensors(R, A, b, boundary)

    # Create the links
    link = [Index(size(tensors[r], 2); tags="link $r") for r in 1:(R - 1)]

    # Fill the MPO, taking care to not include auxiliary links at the edges
    mpo = MPO(R)
    spin_dims = ntuple(_ -> 2, M + N)
    if R == 1
        mpo[1] = ITensor(reshape(tensors[1], spin_dims...),
            (y[1, :]..., x[1, :]...))
    elseif R > 1
        mpo[1] = ITensor(reshape(tensors[1], size(tensors[1], 2), spin_dims...),
            (link[1], y[1, :]..., x[1, :]...))
        for r in 2:(R - 1)
            newshape = size(tensors[r])[1:2]..., spin_dims...
            mpo[r] = ITensor(reshape(tensors[r], newshape),
                (link[r - 1], link[r], y[r, :]..., x[r, :]...))
        end
        mpo[R] = ITensor(reshape(tensors[R], size(tensors[R], 1), spin_dims...),
            (link[R - 1], y[R, :]..., x[R, :]...))
    end
    return mpo
end

"""
    affine_transform_tensors(R, A, b, [boundary])

Compute vector of core tensors (constituent 4-way tensors) for a matrix product
operator corresponding to one of affine transformation `y = A*x + b` with
rational `A` and `b`
"""
function affine_transform_tensors(
        R::Integer, A::AbstractMatrix{<:Union{Integer,Rational}},
        b::AbstractVector{<:Union{Integer,Rational}},
        boundary::AbstractBoundaryConditions=PeriodicBoundaryConditions())
    tensors, carry = affine_transform_tensors(
        Int(R), _affine_static_args(A, b)...; boundary)
    return tensors
end

function affine_transform_tensors(
        R::Int, A::SMatrix{M,N,Int}, b::SVector{M,Int}, s::Int;
        boundary::AbstractBoundaryConditions=PeriodicBoundaryConditions()) where {M,N}
    # Checks
    0 <= R * max(M, N) <= 8 * sizeof(Int) ||
        throw(DomainError(R, "invalid value of the length R"))

    # The output tensors are a collection of matrices, but their first two
    # dimensions (links) vary
    tensors = Vector{Array{Bool,4}}(undef, R)

    # The initial carry is zero
    carry = [zero(SVector{M,Int})]
    for r in R:-1:1
        # Figure out the current bit to add from the shift term and shift
        bcurr = SVector{M,Int}((copysign(b_, abs(b_)) & 1 for b_ in b))

        # Get tensor.
        new_carry, data = affine_transform_core(A, bcurr, s, carry)

        # XXX do pruning: cut away carries that are dead ends in further
        #     tensors

        #if r == 1
        # For the first tensor, we examine the carry to see which elements
        # contribute with which weight
        #weights = map(c -> carry_weight(c, boundary), new_carry)
        #tensors[r] = sum(data .* weights, dims=1)
        #else
        tensors[r] = data
        #end

        # Set carry to the next value
        carry = new_carry
        b = @. b >> 1
    end

    if boundary == OpenBoundaryConditions() && maximum(abs, b) > 0
        # Extend the tensors to the left until we have no more nonzero bits in b
        # This is equivalent to a larger domain.
        tensors_ext = Array{Bool,4}[]
        while maximum(abs, b) > 0
            bcurr = SVector{M,Int}((copysign(b_, abs(b_)) & 1 for b_ in b))
            new_carry, data = affine_transform_core(A, bcurr, s, carry; activebit=false)
            pushfirst!(tensors_ext, data)

            carry = new_carry
            b = @. b >> 1
        end

        weights = map(c -> carry_weight(c, boundary), carry)
        tensors_ext[1] = sum(tensors_ext[1] .* weights; dims=1)
        _matrix(x) = reshape(x, size(x, 1), size(x, 2))
        cap_matrix = reduce(*, _matrix.(tensors_ext))

        tensors[1] = reshape(
            cap_matrix * reshape(tensors[1], size(tensors[1], 1), :),
            size(cap_matrix, 1), size(tensors[1])[2:end]...
        )
    else
        weights = map(c -> carry_weight(c, boundary), carry)
        tensors[1] = sum(tensors[1] .* weights; dims=1)
    end

    return tensors, carry
end

"""
    core, out_carry = affine_transform_core(A, b, s, in_carry)

Construct core tensor `core` for an affine transform.  The core tensor for an
affine transformation is given by:

    core[d, c, iy, ix] =
        2 * out_carry[d] == A * x[ix] + b - s * y[iy] + in_carry[c]

where `A`, a matrix of integers, and `b`, a vector of bits, which define the
affine transform. `c` and `d` are indices into a set of integer vectors
`in_carry` and `out_carry`, respectively, which encode the incoming and outgoing
carry from the other core tensors. `x[ix] ∈ {0,1}^N` and `y[iy] ∈ {0,1}^M`
are the binary input and output vectors, respectively, of the affine transform.
They are indexed in a "little-endian" fashion.
"""
function affine_transform_core(
        A::SMatrix{M,N,Int}, b::SVector{M,Int}, s::Int,
        carry::AbstractVector{SVector{M,Int}}; activebit=true
) where {M,N}

    # Otherwise we have to reverse the indexing of x and y
    M <= N ||
        throw(ArgumentError("expect wide transformation matrix"))

    # The basic idea here is the following: we check
    #
    #           A*x + b - s*y + c == 2*d
    #
    # for all "incoming" carrys c and all possible bit vectors, x ∈ {0,1}^N.
    # and y ∈ {0,1}^M for some outgoing carry d, which may be negative.
    # We then store this as something like out[d, c, x, y].
    out = Dict{SVector{M,Int},Array{Bool,3}}()
    sizehint!(out, length(carry))

    bitrange = activebit ? range(0, 1) : range(0, 0)
    all_x = Iterators.product(ntuple(_ -> bitrange, N)...)
    all_y = Iterators.product(ntuple(_ -> bitrange, M)...)

    for (c_index, c) in enumerate(carry)
        for (x_index, x) in enumerate(all_x)
            z = A * SVector{N,Bool}(x) + b + SVector{M,Int}(c)

            if isodd(s)
                # if s is odd, then there is a unique y which solves satisfies
                # above condition (simply the lowest bit)
                y = @. Bool(z & 1)
                y_index = digits_to_number(y) + 1

                # Correct z and compute carry
                d = @. (z - s * y) .>> 1

                # Store this
                d_mat = get!(out, d) do
                    return zeros(
                        Bool, length(carry), length(bitrange)^M, length(bitrange)^N)
                end
                @inbounds d_mat[c_index, y_index, x_index] = true
            else
                # if s instead even, then the conditions for x and y decouple.
                # since y cannot touch the lowest bit, z must already be even.
                all(iseven, z) ||
                    continue

                # y can take any value at this point. This may lead to
                # bonds that do not contribute because they become dead ends
                # in a subsequent tensor (no valid outgoing carries). We cannot
                # decide this here, but must prune those "branches" from the
                # right once in the driver routine (affine_transform_tensors).
                for (y_index, y) in enumerate(all_y)
                    # Correct z and compute carry
                    d = @. (z - s * y) >> 1

                    # Store this
                    d_mat = get!(out, d) do
                        return zeros(
                            Bool, length(carry), length(bitrange)^M, length(bitrange)^N)
                    end
                    @inbounds d_mat[c_index, y_index, x_index] = true
                end
            end
        end
    end

    # We translate the dictionary into a vector of carrys (which we can then
    # pass into the next iteration) and a 4-way tensor of output values.
    carry_out = Vector{SVector{M,Int}}(undef, length(out))
    #value_out = Array{Bool,4}(undef, length(out), length(carry), 1 << M, 1 << N)
    value_out = Array{Bool,4}(
        undef, length(out), length(carry), length(bitrange)^M, length(bitrange)^N)
    for (p_index, p) in enumerate(pairs(out))
        carry_out[p_index] = p.first
        value_out[p_index, :, :, :] .= p.second
    end
    return carry_out, value_out
end

"""
    affine_transform_matrix(R, A, b, [boundary])

Compute full transformation matrix for the affine transformation `y = A*x + b`,
where `y` is a `M`-vector and `x` is `N`-vector, and each component is in
`{0, 1, ..., 2^R-1}`. `A` is a rational `M × N` matrix and `b` is a rational
`N`-vector.

Return a boolean sparse `2^(R*M) × 2^(R*N)` matrix `A`, which has true entries
whereever the condition is satisfied. The element indices `A[iy, ix]` are
mapped to `x` and `y` as follows:

    iy = 1 + y[1] + y[2] * 2^R + y[3] * 2^(2R) + ... + y[M] * 2^((M-1)*R)
    ix = 1 + x[1] + x[2] * 2^R + x[3] * 2^(2R) + ... + x[N] * 2^((N-1)*R)

`boundary` specifies the type of boundary conditions.
"""
function affine_transform_matrix(
        R::Integer, A::AbstractMatrix{<:Union{Integer,Rational}},
        b::AbstractVector{<:Union{Integer,Rational}},
        boundary::AbstractBoundaryConditions=PeriodicBoundaryConditions()
)
    return affine_transform_matrix(Int(R), _affine_static_args(A, b)..., boundary)
end

function affine_transform_matrix(
        R::Int, A::SMatrix{M,N,Int}, b::SVector{M,Int},
        s::Int, boundary::AbstractBoundaryConditions) where {M,N}
    # Checks
    0 <= R * max(M, N) <= 8 * sizeof(Int) ||
        throw(DomainError(R, "invalid value of the length R"))

    mask = ~(~0 << R)
    y_index = Int[]
    x_index = Int[]
    sizehint!(y_index, 1 << (R * max(M, N)))
    sizehint!(x_index, 1 << (R * max(M, N)))

    all_x = Iterators.product(ntuple(_ -> 0:mask, N)...)
    all_y = Iterators.product(ntuple(_ -> 0:mask, M)...)
    for (ix, x) in enumerate(all_x)
        v = A * SVector{N,Int}(x) + b
        for (iy, y) in enumerate(all_y)
            if equiv(v, s * SVector{M,Int}(y), R, boundary)
                #println("$y <- $x")
                push!(y_index, iy)
                push!(x_index, ix)
            end
        end
    end
    values = ones(Bool, size(x_index))
    return sparse(y_index, x_index, values, 1 << (R * M), 1 << (R * N))
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
        out_indices = vec(reverse(outsite; dims=1))
        in_indices = vec(reverse(insite; dims=1))
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
    denom = lcm(mapreduce(denominator, lcm, A; init=1),
        mapreduce(denominator, lcm, b; init=1))
    Ai = @. Int(denom * A)
    bi = @. Int(denom * b)

    # Construct static matrix
    return SMatrix{M,N,Int}(Ai), SVector{M,Int}(bi), denom
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

    # XXX: we do not support pseudo-inverses (LinearAlgebra cannot do
    #      this yet over the Rationals).
    Tinv = inv(T)
    Ainv = Tinv[1:m, 1:n]
    binv = Tinv[1:m, n + 1]
    return Ainv, binv
end

"""
    digits_to_number(v::AbstractVector{Bool})

Converts a vector of digits, starting with the least significant digit, to
a number.
"""
digits_to_number(v::AbstractVector{Bool}) = _digits_to_number(Tuple(v))

@inline _digits_to_number(v::Tuple{}) = 0
@inline _digits_to_number(v::Tuple{Bool,Vararg{Bool}}) = _digits_to_number(v[2:end]) << 1 |
                                                         v[1]
