
@testitem "grid.jl" begin
    using Test
    import Quantics

    @testset "InherentDiscreteGrid" begin
        m = Quantics.InherentDiscreteGrid{3}(5)
        @test Quantics.grid_min(m) == (1, 1, 1)
        @test Quantics.grid_step(m) == (1, 1, 1)
        for idx in [(1, 1, 1), (1, 1, 2)]
            c = Quantics.grididx_to_origcoord(m, idx)
            @test Quantics.origcoord_to_grididx(m, c) == idx
        end
    end

    @testset "DiscretizedGrid" begin
        @testset "1D" begin
            R = 5
            grid_min = 0.1
            grid_max = 2.0
            dx = (grid_max - grid_min) / 2^R
            g = Quantics.DiscretizedGrid{1}(R, (grid_min,), (grid_max,))
            @test @inferred(Quantics.origcoord_to_grididx(g, 0.999999 * dx + grid_min)) == 1
            @test Quantics.origcoord_to_grididx(g, 1.999999 * dx + grid_min) == 2
            @test Quantics.origcoord_to_grididx(g, grid_max - 1e-9 * dx) == 2^R
            @test Quantics.grid_min(g) == (0.1,)
            @test Quantics.grid_step(g) == (0.059375,)
        end

        @testset "2D" begin
            R = 5
            d = 2
            grid_min = (0.1, 0.1)
            grid_max = (2.0, 2.0)
            dx = (grid_max .- grid_min) ./ 2^R
            g = Quantics.DiscretizedGrid{d}(R, grid_min, grid_max)

            @test Quantics.grid_min(g) == (0.1, 0.1)
            @test Quantics.grid_step(g) == dx ≈ (0.059375, 0.059375)

            cs = [
                0.999999 .* dx .+ grid_min,
                1.999999 .* dx .+ grid_min,
                grid_max .- 1e-9 .* dx
            ]
            refs = [1, 2, 2^R]

            for (c, ref) in zip(cs, refs)
                @inferred(Quantics.origcoord_to_grididx(g, c))
                @test all(Quantics.origcoord_to_grididx(g, c) .== ref)
            end

            @test_throws "Bound Error:" Quantics.origcoord_to_grididx(g, (0.0, 0.0))
            @test_throws "Bound Error:" Quantics.origcoord_to_grididx(g, (0.0, 1.1))
            @test_throws "Bound Error:" Quantics.origcoord_to_grididx(g, (1.1, 0.0))
            @test_throws "Bound Error:" Quantics.origcoord_to_grididx(g, (3.0, 1.1))
            @test_throws "Bound Error:" Quantics.origcoord_to_grididx(g, (1.1, 3.0))
            @test_throws "Bound Error:" Quantics.origcoord_to_grididx(g, (3.0, 3.0))
        end
    end
end
