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
    tensors = affine_transform_matrices(
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

function affine_transform_matrices(
            R::Int, A::SMatrix{M, N, Int}, b::SVector{M, Int}
            ) where {M, N}
    # Checks
    0 <= R <= 62 ||
        throw(ArgumentError("invalid value of the length R"))

    # Matrix which maps out which bits to add together. We use 2's complement,
    # -x == ~x + 1, which means we use two masks: A1 operating on set bits, and
    # A0 operating on unset bits. We group the +1 together with the shift.
    A1 = @. clamp(A, 0, typemax(Int))
    A0 = @. clamp(-A, 0, typemax(Int))
    b = b .+ vec(sum(Bool.(A0), dims=2))
    #println(A1, A0, b)

    # Amax is the maximum value that can be reached by multiplying it with set
    # or unset bits.
    Amax = vec(sum(A0 + A1, dims=2))

    # The output tensors are a collection of matrices, but their first two
    # dimensions (links) vary
    tensors = Array{Bool, 4}[]
    sizehint!(tensors, R)

    # We have to carry all but the least siginificant bit to the adjacent
    # tensor. α is the index of the "outgoing" carrys, which at the beginning
    # is simply all zeros.
    maxcarry_α = zeros(typeof(b))
    ncarry_α = prod(maxcarry_α .+ 1)
    base_α = _indexbase_from_size(maxcarry_α, 1)
    for r in 1:R
        # Copy the outgoing carry α of the previous tensor to the "incoming"
        # carry β of the current tensor.
        maxcarry_β = maxcarry_α
        ncarry_β = ncarry_α

        # Figure out the current bit to add from the shift term and shift
        # it out from the array
        bcurr = @. b & 1
        b = @. b >> 1

        # Determine the maximum outgoing carry. It is determined by the maximum
        # previous carry plus the maximum value of the mapped indices plus the
        # current shift, all divided by two. In the case of the last tensor, we
        # discard all carrys.
        if r == R
            maxcarry_α = zeros(typeof(maxcarry_α))
            ncarry_α = 1
            base_α = _indexbase_from_size(maxcarry_α, 0)
        else
            maxcarry_α =  @. (maxcarry_β + Amax + bcurr) >> 1
            ncarry_α = prod(maxcarry_α .+ 1)
            base_α = _indexbase_from_size(maxcarry_α, 1)
        end

        values = zeros(Bool, ncarry_α, ncarry_β, 1 << M, 1 << N)
        #println("\n r=$r $(bcurr)")
        # Fill values array. The idea is the following: we iterate over all
        # possible values of the carry from the previous tensor (c_β). Each of
        # those corresponds to a bond index β.
        allcarry_β = map(i -> 0:i, maxcarry_β)
        for (β, _c_β) in enumerate(Iterators.product(allcarry_β...))
            c_β = SVector{M}(_c_β)

            # Now we iterate over all possible indices of the input legs, which
            # are all combination of values of N input bits
            allspin_j = ntuple(_ -> 0:1, N)
            for (j, _σ_j) in enumerate(Iterators.product(allspin_j...))
                σ_j = SVector{N, Bool}(_σ_j)

                # We apply the transformation y = Ax + b to the current bits
                # σ_j and adding the incoming carry c_β. The least significant
                # bits are then the output σ_i while the higher-order bits
                # form the outgoing carry c_α
                ifull = A1 * σ_j + A0 * (.~σ_j) + bcurr + c_β
                c_α = @. ifull >> 1
                σ_i = SVector{M, Bool}(@. ifull & 1)

                # Map the outgoing carry to an index and store
                α = mapreduce(*, +, c_α, base_α, init=1)
                i = _spins_to_number(σ_i) + 1
                #println("$(c_α) * 2 + $(σ_i) <- $(c_β) * 2 + $(σ_j)")
                values[α, β, i, j] = true
            end
        end
        push!(tensors, values)
    end
    return tensors
end

"""
    get_int_inverse(A::AbstractMatrix{<:Integer})

Return inverse matrix to integer A, ensuring that it is indeed integer.
"""
function get_int_invserse(A::AbstractMatrix{<:Integer})
    Ainv = inv(A)
    Ainv_int = map(eltype(A) ∘ round, Ainv)
    isapprox(Ainv, Ainv_int, atol=size(A,1)*eps()) ||
        throw(DomainError(Ainv, "inverse is not integer"))
    return typeof(A)(Ainv)
end

_indexbase_from_size(v::SVector{M,T}, init::T) where {M,T} =
    SVector{M,T}(_indexbase_from_size(Tuple(v), init))
_indexbase_from_size(v::Tuple{}, init::T) where {T} = ()
_indexbase_from_size(v::Tuple{T, Vararg{T}}, init::T) where {T} =
    init, _indexbase_from_size(v[2:end], init * v[1])...

_spins_to_number(v::SVector{<:Any, Bool}) = _spins_to_number(Tuple(v))
_spins_to_number(v::Tuple{Bool, Vararg{Bool}}) =
    _spins_to_number(v[2:end]) << 1 | v[1]
_spins_to_number(v::Tuple{}) = 0
