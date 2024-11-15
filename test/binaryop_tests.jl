@testitem "binaryop_tests.jl/_binaryop" begin
    using Test
    using ITensors
    using ITensorMPS: randomMPS
    ITensors.disable_warn_order()
    using Quantics
    import Random

    @testset "_binaryop" for rev_carrydirec in [true], nbit in 2:3
        Random.seed!(1)
        # For a = +/- 1, b = +/- 1, c = +/- 1, d = +/- 1,
        # x' = a * x + b * y
        # y' = c * x + d * y
        # f(x, y) = g(x', y')
        # excluding a + b == -2 || c + d == -2
        if rev_carrydirec
            # x1, y1, x2, y2, ...
            sites = [Index(2, "Qubit, $name=$n") for n in 1:nbit for name in ["x", "y"]]
        else
            # xR, yR, xR-1, yR-1, ...
            sites = [Index(2, "Qubit, $name=$n") for n in reverse(1:nbit)
                     for name in ["x", "y"]]
        end
        # x1, x2, ...
        sitesx = [sites[findfirst(x -> hastags(x, "x=$n"), sites)] for n in 1:nbit]
        # y1, y2, ...
        sitesy = [sites[findfirst(x -> hastags(x, "y=$n"), sites)] for n in 1:nbit]
        rsites = reverse(sites)

        for a in -1:1, b in -1:1, c in -1:1, d in -1:1, bc_x in [1, -1], bc_y in [1, -1]
            g = randomMPS(sites)
            M = Quantics._binaryop_mpo(sites, [(a, b), (c, d)], [(1, 2), (1, 2)];
                rev_carrydirec=rev_carrydirec, bc=[bc_x, bc_y])
            f = apply(M, g)

            # f[x_R, ..., x_1, y_R, ..., y_1] and f[x, y]
            f_arr = Array(reduce(*, f), vcat(reverse(sitesx), reverse(sitesy)))
            f_vec = reshape(f_arr, 2^nbit, 2^nbit)

            # g[x_R, ..., x_1, y_R, ..., y_1] and g[x, y]
            g_arr = Array(reduce(*, g), vcat(reverse(sitesx), reverse(sitesy)))
            g_vec = reshape(g_arr, 2^nbit, 2^nbit)

            function prime_xy(x, y)
                0 <= x < 2^nbit || error("something went wrong")
                0 <= y < 2^nbit || error("something went wrong")
                xp_ = a * x + b * y
                yp_ = c * x + d * y
                nmodx, xp = divrem(xp_, 2^nbit, RoundDown)
                nmody, yp = divrem(yp_, 2^nbit, RoundDown)
                return xp, yp, bc_x^nmodx, bc_y^nmody
            end

            f_vec_ref = similar(f_vec)
            for x in 0:(2^nbit - 1), y in 0:(2^nbit - 1)
                xp, yp, sign_x, sign_y = prime_xy(x, y)
                f_vec_ref[x + 1, y + 1] = g_vec[xp + 1, yp + 1] * sign_x * sign_y
            end

            @test f_vec_ref ≈ f_vec
        end
    end
end

@testitem "binaryop_tests.jl/affinetransform" begin
    using Test
    using ITensors
    using ITensorMPS: randomMPS
    ITensors.disable_warn_order()
    using Quantics
    import Random

    @testset "affinetransform" for rev_carrydirec in [true, false], nbit in 2:3
        Random.seed!(1)
        # For a, b, c, d = +1, -1, 0,
        #   x' = a * x + b * y + s1
        #   y' = c * x + d * y + s2
        # f(x, y) = g(x', y')
        if rev_carrydirec
            # x1, y1, x2, y2, ...
            sites = [Index(2, "Qubit, $name=$n") for n in 1:nbit for name in ["x", "y"]]
        else
            # xR, yR, xR-1, yR-1, ...
            sites = [Index(2, "Qubit, $name=$n") for n in reverse(1:nbit)
                     for name in ["x", "y"]]
        end
        # x1, x2, ...
        sitesx = [sites[findfirst(x -> hastags(x, "x=$n"), sites)] for n in 1:nbit]
        # y1, y2, ...
        sitesy = [sites[findfirst(x -> hastags(x, "y=$n"), sites)] for n in 1:nbit]
        shift = rand((-2 * 2^nbit):(2 * 2^nbit), 2)

        for a in -1:1, b in -1:1, c in -1:1, d in -1:1, bc_x in [1, -1], bc_y in [1, -1]
            g = randomMPS(sites)
            f = Quantics.affinetransform(g, ["x", "y"],
                [Dict("x" => a, "y" => b), Dict("x" => c, "y" => d)],
                shift, [bc_x, bc_y]; cutoff=1e-25)

            # f[x_R, ..., x_1, y_R, ..., y_1] and f[x, y]
            f_arr = Array(reduce(*, f), vcat(reverse(sitesx), reverse(sitesy)))
            f_vec = reshape(f_arr, 2^nbit, 2^nbit)

            # g[x_R, ..., x_1, y_R, ..., y_1] and g[x, y]
            g_arr = Array(reduce(*, g), vcat(reverse(sitesx), reverse(sitesy)))
            g_vec = reshape(g_arr, 2^nbit, 2^nbit)

            function prime_xy(x, y)
                0 <= x < 2^nbit || error("something went wrong")
                0 <= y < 2^nbit || error("something went wrong")
                xp_ = a * x + b * y + shift[1]
                yp_ = c * x + d * y + shift[2]
                nmodx, xp = divrem(xp_, 2^nbit, RoundDown)
                nmody, yp = divrem(yp_, 2^nbit, RoundDown)
                return xp, yp, bc_x^nmodx, bc_y^nmody
            end

            f_vec_ref = similar(f_vec)
            for x in 0:(2^nbit - 1), y in 0:(2^nbit - 1)
                xp, yp, sign_x, sign_y = prime_xy(x, y)
                f_vec_ref[x + 1, y + 1] = g_vec[xp + 1, yp + 1] * sign_x * sign_y
            end

            @test f_vec_ref ≈ f_vec
        end
    end
end

@testitem "binaryop_tests.jl/affinetransform_three_vars" begin
    using Test
    using ITensors
    using ITensorMPS: randomMPS
    ITensors.disable_warn_order()
    using Quantics
    import Random

    affinetransform_testsets = []

    # x' = x + y
    # y' =     y + z
    # z' = x     + z
    push!(affinetransform_testsets, [1 1 0; 0 1 1; 1 0 1])

    # x' = -x - y
    # y' =      y + z
    # z' =  x     + z
    push!(affinetransform_testsets, [-1 -1 0; 0 1 1; 1 0 1])

    # x' = -x     + z
    # y' =      y + z
    # z' =  x     + z
    push!(affinetransform_testsets, [-1 0 1; 0 1 1; 1 0 1])

    # x' =      y + z
    # y' =      y + z
    # z' =  x     + z
    push!(affinetransform_testsets, [0 1 1; 0 1 1; 1 0 1])

    # x' =      y - z
    # y' =      y
    # z' =  x
    push!(affinetransform_testsets, [0 1 -1; 0 1 0; 1 0 0])

    @testset "affinetransform_three_vars" for rev_carrydirec in [true, false],
        bc_x in [1, -1], bc_y in [1, -1],
        bc_z in [1, -1], nbit in 2:3,
        affmat in affinetransform_testsets
        #@testset "affinetransform_three_var" for rev_carrydirec in [true], bc_x in [1], bc_y in [1], bc_z in [1], nbit in [2], affmat in affinetransform_testsets
        Random.seed!(1234)
        varnames = ["x", "y", "z", "K"] # "K" is not involved in transform

        # Read coefficient matrix
        coeffs_dic = Dict{String,Int}[]
        for newvar in 1:3
            @test all(abs.(affmat[newvar, :]) .<= 1)
            @test sum(abs.(affmat[newvar, :])) <= 2
            @test sum(abs.(affmat[newvar, :])) > 0

            coeffs = Dict{String,Int}()
            for oldvar in 1:3
                if affmat[newvar, oldvar] != 0
                    coeffs[varnames[oldvar]] = affmat[newvar, oldvar]
                end
            end
            if length(coeffs) == 1
                for oldvar in 1:3
                    if !(varnames[oldvar] ∈ keys(coeffs))
                        coeffs[varnames[oldvar]] = 0
                        break
                    end
                end
            end
            @test length(coeffs) == 2
            push!(coeffs_dic, coeffs)
        end

        if rev_carrydirec
            # x1, y1, z1, x2, y2, z2, ...
            sites = [Index(2, "Qubit, $name=$n") for n in 1:nbit for name in varnames]
        else
            # xR, yR, zR, xR-1, yR-1, zR-1...
            sites = [Index(2, "Qubit, $name=$n") for n in reverse(1:nbit)
                     for name in varnames]
        end
        # x1, x2, ...
        sitesx = [sites[findfirst(x -> hastags(x, "x=$n"), sites)] for n in 1:nbit]
        # y1, y2, ...
        sitesy = [sites[findfirst(x -> hastags(x, "y=$n"), sites)] for n in 1:nbit]
        # z1, z2, ...
        sitesz = [sites[findfirst(x -> hastags(x, "z=$n"), sites)] for n in 1:nbit]
        # K1, K2, ...
        sitesK = [sites[findfirst(x -> hastags(x, "K=$n"), sites)] for n in 1:nbit]

        shift = rand((-2 * 2^nbit):(2 * 2^nbit), 3)

        g = randomMPS(sites)
        f = Quantics.affinetransform(g, ["x", "y", "z"],
            coeffs_dic,
            shift, [bc_x, bc_y, bc_z]; cutoff=1e-25)

        # f[x_R, ..., x_1, y_R, ..., y_1, z_R, ..., z_1] and f[x, y, z]
        f_arr = Array(reduce(*, f),
            vcat(reverse(sitesx), reverse(sitesy), reverse(sitesz),
                reverse(sitesK)))
        f_vec = reshape(f_arr, 2^nbit, 2^nbit, 2^nbit, 2^nbit)

        # g[x'_R, ..., x'_1, y'_R, ..., y'_1, z'_R, ..., z'_1] and g[x', y', z']
        g_arr = Array(reduce(*, g),
            vcat(reverse(sitesx), reverse(sitesy), reverse(sitesz),
                reverse(sitesK)))
        g_vec = reshape(g_arr, 2^nbit, 2^nbit, 2^nbit, 2^nbit)

        function prime_xy(x, y, z)
            xp_, yp_, zp_ = affmat * [x, y, z] .+ shift
            nmodx, xp = divrem(xp_, 2^nbit, RoundDown)
            nmody, yp = divrem(yp_, 2^nbit, RoundDown)
            nmodz, zp = divrem(zp_, 2^nbit, RoundDown)
            return xp, yp, zp, bc_x^nmodx, bc_y^nmody, bc_z^nmodz
        end

        f_vec_ref = similar(f_vec)
        for x in 0:(2^nbit - 1), y in 0:(2^nbit - 1), z in 0:(2^nbit - 1)
            xp, yp, zp, sign_x, sign_y, sign_z = prime_xy(x, y, z)
            f_vec_ref[x + 1, y + 1, z + 1, :] .= g_vec[xp + 1, yp + 1, zp + 1, :] * sign_x *
                                                 sign_y * sign_z
        end

        @test f_vec_ref ≈ f_vec
    end
end

@testitem "binaryop_tests.jl/shiftop" begin
    using Test
    using ITensors
    using ITensorMPS: randomMPS
    ITensors.disable_warn_order()
    using Quantics
    import Random

    @testset "shiftop" for R in [3], bc in [1, -1]
        sites = [Index(2, "Qubit, x=$n") for n in 1:R]
        g = randomMPS(sites)

        for shift in [0, 1, 2, 2^R - 1]
            M = Quantics._shift_mpo(sites, shift; bc=bc)
            f = apply(M, g)

            f_vec = vec(Array(reduce(*, f), reverse(sites)))
            g_vec = vec(Array(reduce(*, g), reverse(sites)))

            f_vec_ref = similar(f_vec)
            for i in 1:(2^R)
                ishifted = mod1(i + shift, 2^R)
                sign = ishifted == i + shift ? 1 : bc
                f_vec_ref[i] = g_vec[ishifted] * sign
            end

            @test f_vec_ref ≈ f_vec
        end
    end
end
