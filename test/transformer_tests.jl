
@testitem "transformer_tests.jl/functions" begin
    using Test
    import Quantics
    using ITensors
    using ITensors.SiteTypes: siteinds
    using ITensorMPS: randomMPS
    using LinearAlgebra

    @testset "upper_lower_triangle" for upper_or_lower in [:upper, :lower]
        R = 3
        sites = siteinds("Qubit", R)
        trimat = Quantics.upper_lower_triangle_matrix(sites, 1.0;
            upper_or_lower=upper_or_lower)
        trimatdata = Array(reduce(*, trimat), [reverse(sites')..., reverse(sites)...])
        trimatdata = reshape(trimatdata, 2^R, 2^R)

        ref = upper_or_lower == :lower ? [Float64(i > j) for i in 1:(2^R), j in 1:(2^R)] :
              [Float64(i < j) for i in 1:(2^R), j in 1:(2^R)]

        @test trimatdata ≈ ref
    end

    @testset "cusum" begin
        R = 3
        sites = siteinds("Qubit", R)
        UT = Quantics.upper_lower_triangle_matrix(sites, 1.0; upper_or_lower=:lower)
        f = Quantics.expqtt(sites, -1.0)
        f_values = vec(Array(reduce(*, f), reverse(sites)))
        xs = collect(LinRange(0, 1, 2^R + 1)[1:(end - 1)])

        g = apply(UT, f)
        g_values = vec(Array(reduce(*, g), reverse(sites)))

        g_values_ref = cumsum(f_values) .- f_values # Second term remove the own values

        @test g_values ≈ g_values_ref
    end

    @testset "flipop" for nbit in 2:3, rev_carrydirec in [true, false], bc in [1, -1]
        sites = siteinds("Qubit", nbit)

        g = randomMPS(rev_carrydirec ? sites : reverse(sites))

        op = Quantics.flipop(siteinds(g); rev_carrydirec=rev_carrydirec, bc=bc)
        f = apply(op, g; alg="naive")
        g_reconst = vec(Array(reduce(*, g), reverse(sites)))
        f_reconst = vec(Array(reduce(*, f), reverse(sites)))

        f_ref = similar(f_reconst)
        for i in 0:(2^nbit - 1)
            nmod, i_ = divrem(2^nbit - i, 2^nbit, RoundDown)
            f_ref[i + 1] = g_reconst[i_ + 1] * (bc^nmod)
        end

        @test f_reconst ≈ f_ref
    end
end

@testitem "transformer_tests.jl/reverseaxis" begin
    using Test
    import Quantics
    using ITensors
    using ITensorMPS: randomMPS
    using LinearAlgebra

    @testset "reverseaxis" for bc in [1], nbit in 2:2, rev_carrydirec in [true, false]
        sitesx = [Index(2, "x=$x") for x in 1:nbit]
        sites = rev_carrydirec ? sitesx : reverse(sitesx)

        g = randomMPS(sites)

        f = Quantics.reverseaxis(g; tag="x", alg="naive", bc=bc)
        g_reconst = vec(Array(reduce(*, g), reverse(sitesx)))
        f_reconst = vec(Array(reduce(*, f), reverse(sitesx)))

        f_ref = similar(f_reconst)
        for i in 1:(2^nbit)
            f_ref[i] = g_reconst[mod(2^nbit - (i - 1), 2^nbit) + 1]
        end
        f_ref[1] *= bc

        @test f_reconst ≈ f_ref
    end

    @testset "reverseaxis2" for nbit in 2:3, rev_carrydirec in [true, false]
        N = 2^nbit

        sitesx = [Index(2, "x=$x") for x in 1:nbit]
        sitesy = [Index(2, "y=$y") for y in 1:nbit]

        if rev_carrydirec
            sites = collect(Iterators.flatten(zip(sitesx, sitesy)))
        else
            sites = collect(Iterators.flatten(zip(reverse(sitesx), reverse(sitesy))))
        end

        g = randomMPS(sites)

        function _reconst(M)
            arr = Array(reduce(*, M), [reverse(sitesx)..., reverse(sitesy)...])
            return reshape(arr, N, N)
        end

        g_reconst = _reconst(g)

        fx = Quantics.reverseaxis(g; tag="x", alg="naive")
        fx_reconst = _reconst(fx)

        fy = Quantics.reverseaxis(g; tag="y", alg="naive")
        fy_reconst = _reconst(fy)

        fx_ref = similar(fx_reconst)
        fy_ref = similar(fy_reconst)
        for j in 0:(N - 1), i in 0:(N - 1)
            fx_ref[i + 1, j + 1] = g_reconst[mod(N - i, N) + 1, j + 1]
            fy_ref[i + 1, j + 1] = g_reconst[i + 1, mod(N - j, N) + 1]
        end

        @test fx_reconst ≈ fx_ref
        @test fy_reconst ≈ fy_ref
    end

    #@testset "reverseaxis3" for nbit in 2:3, rev_carrydirec in [true, false]
    @testset "reverseaxis3" for nbit in [2], rev_carrydirec in [true]
        N = 2^nbit

        sitesx = [Index(2, "x=$x") for x in 1:nbit]
        sitesy = [Index(2, "y=$y") for y in 1:nbit]
        sitesz = [Index(2, "z=$z") for z in 1:nbit]

        if rev_carrydirec
            sites = collect(Iterators.flatten(zip(sitesx, sitesy, sitesz)))
        else
            sites = collect(Iterators.flatten(zip(reverse(sitesx), reverse(sitesy),
                reverse(sitesz))))
        end

        g = randomMPS(sites)

        function _reconst(M)
            arr = Array(reduce(*, M),
                [reverse(sitesx)..., reverse(sitesy)..., reverse(sitesz)...])
            return reshape(arr, N, N, N)
        end

        g_reconst = _reconst(g)

        fx = Quantics.reverseaxis(g; tag="x", alg="naive")
        fx_reconst = _reconst(fx)

        fy = Quantics.reverseaxis(g; tag="y", alg="naive")
        fy_reconst = _reconst(fy)

        fz = Quantics.reverseaxis(g; tag="z", alg="naive")
        fz_reconst = _reconst(fz)

        fx_ref = similar(fx_reconst)
        fy_ref = similar(fy_reconst)
        fz_ref = similar(fz_reconst)
        for k in 0:(N - 1), j in 0:(N - 1), i in 0:(N - 1)
            fx_ref[i + 1, j + 1, k + 1] = g_reconst[mod(N - i, N) + 1, j + 1, k + 1]
            fy_ref[i + 1, j + 1, k + 1] = g_reconst[i + 1, mod(N - j, N) + 1, k + 1]
            fz_ref[i + 1, j + 1, k + 1] = g_reconst[i + 1, j + 1, mod(N - k, N) + 1]
        end

        @test fx_reconst ≈ fx_ref
        @test fy_reconst ≈ fy_ref
        @test fz_reconst ≈ fz_ref
    end

    #==
    @testset "asdiagonal" begin
        R = 2
        sites = siteinds("Qubit", R)
        sites′ = [Index(2, "Qubit,n′=$n") for n in 1:R]

        M = randomMPS(sites)

        for which_new in ["left", "right"]
            Mnew = Quantics.asdiagonal(M, sites′; tag="n", which_new=which_new)

            M_reconst = reshape(Array(reduce(*, M), reverse(sites)), 2^R)
            Mnew_reconst = reshape(Array(reduce(*, Mnew),
                                         vcat(reverse(sites), reverse(sites′))), 2^R, 2^R)

            @assert diag(Mnew_reconst) ≈ M_reconst
            @assert LinearAlgebra.diagm(M_reconst) ≈ Mnew_reconst
        end
    end
    ==#
end

@testitem "transformer_tests.jl/phase_rotation" begin
    using Test
    import Quantics
    using ITensors
    using ITensorMPS: randomMPS
    using LinearAlgebra

    @testset "phase_rotation" begin
        nqbit = 3
        xvec = collect(0:(2^nqbit - 1))
        θ = 0.1
        sites = [Index(2, "Qubit,x=$x") for x in 1:nqbit]
        _reconst(x) = vec(Array(reduce(*, x), reverse(sites)))

        f = randomMPS(sites)
        f_vec = _reconst(f)

        ref = exp.(im * θ * xvec) .* f_vec

        @test ref ≈ _reconst(Quantics.phase_rotation(f, θ; tag="x"))
        @test ref ≈ _reconst(Quantics.phase_rotation(f, θ; targetsites=sites))
    end
end

@testitem "transformer_tests.jl/shiftaxis" begin
    using Test
    import Quantics
    using ITensors
    using ITensorMPS: randomMPS
    using LinearAlgebra

    @testset "shiftaxis" for R in [3], bc in [1, -1], rev_carrydirec in [true, false]
        sitesx = [Index(2, "Qubit, x=$n") for n in 1:R]
        sites = rev_carrydirec ? sitesx : reverse(sitesx)
        g = randomMPS(sites)

        for shift in [0, 1, 2, 2^R - 1]
            f = Quantics.shiftaxis(g, shift; bc=bc, tag="x")

            f_vec = vec(Array(reduce(*, f), reverse(sitesx)))
            g_vec = vec(Array(reduce(*, g), reverse(sitesx)))

            f_vec_ref = similar(f_vec)
            for i in 1:(2^R)
                ishifted = mod1(i + shift, 2^R)
                sign = ishifted == i + shift ? 1 : bc
                f_vec_ref[i] = g_vec[ishifted] * sign
            end

            @test f_vec_ref ≈ f_vec
        end
    end

    @testset "shiftaxis2d" for R in [3], bc in [1, -1], rev_carrydirec in [true, false]
        sitesx = [Index(2, "Qubit, x=$n") for n in 1:R]
        sitesy = [Index(2, "Qubit, y=$n") for n in 1:R]
        if rev_carrydirec
            sites = collect(Iterators.flatten(zip(sitesx, sitesy)))
        else
            sites = collect(Iterators.flatten(zip(reverse(sitesx), reverse(sitesy))))
        end
        g = randomMPS(sites)

        for shift in [-4^R + 1, -1, 0, 1, 2^R - 1, 2^R, 2^R + 1, 4^R + 1]
            f = Quantics.shiftaxis(g, shift; tag="x", bc=bc)

            f_mat = reshape(Array(reduce(*, f), vcat(reverse(sitesx), reverse(sitesy))),
                2^R, 2^R)
            g_mat = reshape(Array(reduce(*, g), vcat(reverse(sitesx), reverse(sitesy))),
                2^R, 2^R)

            f_mat_ref = similar(f_mat)
            for i in 1:(2^R)
                nbc, ishifted = divrem(i + shift - 1, 2^R, RoundDown)
                ishifted += 1
                f_mat_ref[i, :] = g_mat[ishifted, :] * (bc^nbc)
            end

            @test f_mat_ref ≈ f_mat
        end
    end
end
