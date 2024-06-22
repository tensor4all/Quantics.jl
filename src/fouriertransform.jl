@doc raw"""
Create a MPO for Fourier transform

We define two integers using the binary format: ``x = (x_1 x_2 ...., x_N)_2``, ``y = (y_1 y_2 ...., y_N)_2``,
where the right most digits are the least significant digits.

Our definition of the Fourier transform is

```math
    Y(y) = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} X(x) e^{s i \frac{2\pi y x}{N}} = \sum_{x=0}^{N-1} T(y, x) X(x),
```

where we define the transformation matrix ``T`` and ``s = \pm 1``.

The created MPO can transform an input MPS as follows.
We denote the input and output MPS's by ``X`` and ``Y``, respectively.

* ``X(x_1, ..., x_N) = X_1(x_1) ... X_N (x_N)``,
* ``Y(y_N, ..., y_1) = Y_1(y_N) ... Y_N (y_1)``.

"""
function _qft(sites; cutoff::Float64=1e-14, sign::Int=1)
    if any([!hastags(inds(s), "Qubit") for s in sites])
        error("All siteinds for qft must has Qubit tag")
    end
    M = _qft_wo_norm(sites; cutoff=cutoff, sign=sign)
    M *= 2.0^(-0.5 * length(sites))

    # Quick hack: In the Markus's note,
    # the digits are ordered oppositely from the present convention.
    M = MPO([M[n] for n in length(M):-1:1])
    _replace_mpo_siteinds!(M, reverse(sites), sites)

    return M
end

function _assign!(M::MPO, n::Int, arr; autoreshape=false)
    if autoreshape
        arr = reshape(arr, map(dim, inds(M[n]))...)
    end
    M[n] = ITensor(arr, inds(M[n])...)
    return nothing
end

"""
For length(sites) == 1
The resultant MPO is NOT renormalized.
"""
function _qft_nsite1_wo_norm(sites; sign::Int=1)
    length(sites) == 1 || error("num sites > 1")
    _exp(x, k) = exp(sign * im * π * (x - 1) * (k - 1))

    arr = zeros(ComplexF64, 2, 2)
    for out in 1:2, in in 1:2
        arr[out, in] = _exp(out, in)
    end

    M = Quantics._zero_mpo(sites)
    _assign!(M, 1, arr)

    return M
end

function _qft_wo_norm(sites; cutoff::Float64=1e-14, sign::Int=1)
    N = length(sites)
    if N == 1
        return _qft_nsite1_wo_norm(sites; sign=sign)
    end

    M_prev = _qft_wo_norm(sites[2:end]; cutoff=cutoff, sign=sign)
    M_top = _qft_toplayer(sites; sign=sign)

    M = _contract(M_top, M_prev)
    ITensors.truncate!(M; cutoff=cutoff)

    return M
end

function _qft_toplayer(sites; sign::Int=1)
    N = length(sites)
    N > 1 || error("N must be greater than 1")

    tensors = []

    # site = 1
    arr = zeros(ComplexF64, 2, 2, 2)
    for x in 1:2, k in 1:2
        # arr: (out, in, link)
        arr[x, k, k] = exp(sign * im * π * (x - 1) * (k - 1))
    end
    push!(tensors, arr)

    for n in 2:N
        ϕ = π * 0.5^(n - 1)
        _exp(x, k) = exp(sign * im * ϕ * (x - 1) * (k - 1))
        # Right most tensor
        if n == N
            # arr: (link, out, in)
            arr = zeros(ComplexF64, 2, 2, 2)
            for x in 1:2, k in 1:2
                arr[k, x, x] = _exp(x, k)
            end
            push!(tensors, arr)
        else
            # arr: (link_left, out, in, link_right)
            arr = zeros(ComplexF64, 2, 2, 2, 2)
            for x in 1:2, k in 1:2
                arr[k, x, x, k] = _exp(x, k)
            end
            push!(tensors, arr)
        end
    end

    M = Quantics._zero_mpo(sites; linkdims=fill(2, N - 1))
    for n in 1:N
        _assign!(M, n, tensors[n])
    end

    return M
end

function _contract(M_top, M_prev)
    length(M_top) == length(M_prev) + 1 || error("Length mismatch")
    N = length(M_top)
    M_top = ITensors.replaceprime(M_top, 1 => 2; tags="Qubit")
    M_top = ITensors.replaceprime(M_top, 0 => 1; tags="Qubit")
    M_top_ = ITensors.data(M_top)
    M_prev_ = ITensors.data(M_prev)

    M_data = [M_top_[1]]
    for n in 1:(N - 1)
        push!(M_data, M_top_[n + 1] * M_prev_[n])
    end

    M = MPO(M_data)
    M = ITensors.replaceprime(M, 1 => 0; tags="Qubit")
    M = ITensors.replaceprime(M, 2 => 1; tags="Qubit")

    return M
end

abstract type AbstractFT end

struct FTCore
    forward::MPO

    function FTCore(sites; kwargs...)
        new(_qft(sites; kwargs...))
    end
end

nbit(ft::AbstractFT) = length(ft.ftcore.forward)

@doc raw"""
sites[1] corresponds to the most significant digit.
sign = 1

```math
    Y(y) = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} X(x) e^{s i \frac{2\pi (y + y0) (x + x0)}{N}},
```

"""
function forwardmpo(ftcore::FTCore, sites)
    M = copy(ftcore.forward)
    _replace_mpo_siteinds!(M, _extractsites(M), sites)
    return M
end

function backwardmpo(ftcore::FTCore, sites)
    M = conj(MPO(reverse([x for x in ftcore.forward])))
    _replace_mpo_siteinds!(M, _extractsites(M), sites)
    return M
end

function _apply_qft(M::MPO, gsrc::MPS, target_sites, sitepos, sitesdst; kwargs...)
    _replace_mpo_siteinds!(M, _extractsites(M), target_sites)
    M = matchsiteinds(M, siteinds(gsrc))
    gdst = ITensors.apply(M, gsrc; kwargs...)

    N = length(target_sites)
    for n in eachindex(target_sites)
        replaceind!(gdst[sitepos[n]], target_sites[n], sitesdst[N - n + 1])
    end

    return gdst
end

@doc raw"""
Perform Fourier transform for a subset of qubit indices.

We define two integers using the binary format: ``x = (x_1 x_2 ...., x_R)_2``, ``y = (y_1 y_2 ...., y_R)_2``,
where the right most digits are the least significant digits.

The variable `x` is denoted as `src` (source), and the variable `y` is denoted as `dst` (destination).

Our definition of the Fourier transform is

```math
    Y(y) = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} X(x) e^{s i \frac{2\pi (y + y_0) (x + x_0)}{N}}
```

where ``s = \pm 1``, ``x_0`` and ``y_0`` are constants, ``N=2^R``.

`sitessrc[1]` / `sitessrc[end]` corresponds to the most/least significant digit of the input.
`sitesdst[1]` / `sitesdst[end]` corresponds to the most/least significant digit of the output.

`siteinds(M)` must contain `sitessrc` in ascending or descending order.
Instead of specifying `sitessrc`, one can specify the source sites by setting `tag`.
If `tag` = `x`, all sites with tags `x=1`, `x=2`, ... are used as `sitessrc`.
"""
function fouriertransform(M::MPS;
        sign::Int=1,
        tag::String="",
        sitessrc=nothing,
        sitesdst=nothing,
        originsrc::Float64=0.0,
        origindst::Float64=0.0,
        cutoff_MPO=1e-25, kwargs...)
    sites = siteinds(M)
    sitepos, target_sites = _find_target_sites(M; sitessrc=sitessrc, tag=tag)

    if sitesdst === nothing
        sitesdst = target_sites
    end

    if length(target_sites) <= 1
        error("Invalid target_sites")
    end

    # Prepare MPO for QFT
    MQ_ = _qft(target_sites; sign=sign, cutoff=cutoff_MPO)
    MQ = matchsiteinds(MQ_, sites)

    # Phase shift from origindst
    M_result = phase_rotation(M, sign * 2π * origindst / (2.0^length(sitepos));
        targetsites=target_sites)

    # Apply QFT
    M_result = apply(MQ, M_result; kwargs...)

    N = length(target_sites)
    for n in eachindex(target_sites)
        replaceind!(M_result[sitepos[n]], target_sites[n], sitesdst[N - n + 1])
    end

    # Phase shift from originsrc
    M_result = phase_rotation(M_result, sign * 2π * originsrc / (2.0^length(sitepos));
        targetsites=sitesdst)

    M_result *= exp(sign * im * 2π * originsrc * origindst / 2.0^length(sitepos))

    return M_result
end
