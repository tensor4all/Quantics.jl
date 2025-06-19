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
function _qft(sites; cutoff::Float64=1e-25, sign::Int=1)
    if !all(dim.(sites) .== 2)
        error("All siteinds for qft must has Qubit tag")
    end

    R = length(sites)
    R > 1 || error("The number of bits must be greater than 1")

    sites_MPO = collect.(zip(prime.(sites), sites))
    fouriertt = QuanticsTCI.quanticsfouriermpo(R; sign=Float64(sign), normalize=true)
    M = MPO(fouriertt; sites=sites_MPO)

    return truncate(M; cutoff)
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
    gdst = _apply(M, gsrc; kwargs...)

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
        originsrc::Real=0.0,
        origindst::Real=0.0,
        cutoff_MPO=1e-25, kwargs...)
    sites = siteinds(M)
    sitepos, target_sites = _find_target_sites(M; sitessrc=sitessrc, tag=tag)

    if sitesdst === nothing
        sitesdst = target_sites
    end

    if length(target_sites) <= 1
        error("Invalid target_sites")
    end

    p_back = precision(BigFloat)
    setprecision(BigFloat, 256)

    # Prepare MPO for QFT
    MQ_ = _qft(target_sites; sign=sign, cutoff=cutoff_MPO)
    MQ = matchsiteinds(MQ_, sites)

    # Phase shift from origindst
    M_result = phase_rotation(M, sign * 2π * BigFloat(origindst) / (BigFloat(2)^length(sitepos));
        targetsites=target_sites, kwargs...)

    # Apply QFT
    M_result = _apply(MQ, M_result; kwargs...)

    N = length(target_sites)
    for n in eachindex(target_sites)
        replaceind!(M_result[sitepos[n]], target_sites[n], sitesdst[N - n + 1])
    end

    # Phase shift from originsrc
    M_result = phase_rotation(M_result, sign * 2π * BigFloat(originsrc) / (BigFloat(2)^length(sitepos));
        targetsites=sitesdst, kwargs...)

    tmp = Float64(mod(sign * 2π * BigFloat(originsrc) * BigFloat(origindst) / BigFloat(2)^length(sitepos), 2 * π))
    M_result *= exp(im * tmp)

    setprecision(BigFloat, p_back)

    return M_result
end
