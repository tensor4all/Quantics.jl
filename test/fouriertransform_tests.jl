
@testitem "fouriertransform_tests.jl/qft_mpo" begin
    using Test
    using Quantics
    using ITensors
    using ITensorMPS: randomMPS, MPO
    using ITensors.SiteTypes: siteinds
    # A brute-force implementation of _qft (only for tests)
    function _qft_ref(sites; cutoff::Float64=1e-14, sign::Int=1)
        abs(sign) == 1 || error("sign must either 1 or -1")

        nbit = length(sites)
        N = 2^nbit
        sites = noprime(sites)

        tmat = zeros(ComplexF64, N, N)
        for t in 0:(N - 1), x in 0:(N - 1)
            tmat[t + 1, x + 1] = exp(sign * im * 2π * t * x / N)
        end

        # `tmat`: (y_1, ..., y_N, x_1, ..., x_N)
        tmat ./= sqrt(N)
        tmat = reshape(tmat, ntuple(x -> 2, 2 * nbit))

        trans_t = ITensor(tmat, reverse(sites)..., prime(sites)...)
        M = MPO(trans_t, sites; cutoff=cutoff)
        return M
    end

    @testset "qft_mpo" for sign in [1, -1], nbit in [2, 3]
        N = 2^nbit

        sites = siteinds("Qubit", nbit)
        M = Quantics._qft(sites; sign=sign)
        M_ref = _qft_ref(sites; sign=sign)

        @test M ≈ M_ref
    end
end

@testitem "fouriertransform_tests.jl/fouriertransform" begin
    using Test
    using Quantics
    using ITensors
    using ITensorMPS: randomMPS

    function _ft_1d_ref(X, sign, originx, origink)
        N = length(X)
        Y = zeros(ComplexF64, N)
        for k in 1:N
            for x in 1:N
                Y[k] += exp(sign * im * 2π * (k + origink - 1) * (x + originx - 1) / N) *
                        X[x]
            end
        end
        Y ./= sqrt(N)
        return Y
    end

    @testset "fouriertransform_1d" for sign in [1, -1], nbit in [2, 3, 4], originx in [0.1],
        originy in [-0.2]

        N = 2^nbit

        sitesx = [Index(2, "Qubit,x=$x") for x in 1:nbit]
        sitesk = [Index(2, "Qubit,k=$k") for k in 1:nbit]

        # X(x)
        X = randomMPS(sitesx)
        X_vec = Array(reduce(*, X), reverse(sitesx))

        # Y(k)
        Y = Quantics.fouriertransform(X; sign=sign, tag="x", sitesdst=sitesk,
            originsrc=originx, origindst=originy)

        Y_vec_ref = _ft_1d_ref(X_vec, sign, originx, originy)
        Y_vec = vec(Array(reduce(*, Y), reverse(sitesk)))

        @test Y_vec ≈ Y_vec_ref
    end

    function _ft_2d_ref(F::Matrix, sign)
        N = size(F, 1)
        G = zeros(ComplexF64, N, N)
        for ky in 1:N, kx in 1:N
            for y in 1:N, x in 1:N
                G[kx, ky] += exp(sign * im * 2π * (kx - 1) * (x - 1) / N) *
                             exp(sign * im * 2π * (ky - 1) * (y - 1) / N) * F[x, y]
            end
        end
        G ./= N
        return G
    end

    @testset "fouriertransform_2d" for sign in [1, -1], nbit in [2, 3]
        N = 2^nbit

        sitesx = [Index(2, "Qubit,x=$x") for x in 1:nbit]
        sitesy = [Index(2, "Qubit,y=$y") for y in 1:nbit]
        siteskx = [Index(2, "Qubit,kx=$kx") for kx in 1:nbit]
        sitesky = [Index(2, "Qubit,ky=$ky") for ky in 1:nbit]

        sitesin = collect(Iterators.flatten(zip(sitesx, sitesy)))

        # F(x, y)
        # F(x_1, y_1, ..., x_R, y_R)
        F = randomMPS(sitesin)
        F_mat = reshape(Array(reduce(*, F), vcat(reverse(sitesx), reverse(sitesy))), N, N)

        # G(kx, ky)
        # G(kx_R, ky_R, ..., kx_1, ky_1)
        G_ = Quantics.fouriertransform(F; sign=sign, tag="x", sitesdst=siteskx)
        G = Quantics.fouriertransform(G_; sign=sign, tag="y", sitesdst=sitesky)

        G_mat_ref = _ft_2d_ref(F_mat, sign)
        G_mat = reshape(Array(reduce(*, G), vcat(reverse(siteskx), reverse(sitesky))), N, N)

        @test G_mat ≈ G_mat_ref
    end
end
