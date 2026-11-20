@testitem "affine" begin
    using Test
    using ITensors
    using Quantics
    using LinearAlgebra

    # Test results of affine_transform_matrix()
    function test_affine_transform_matrix_multi_variables(R, A, b, T, boundary)
        M, N = size(A)

        yranges = [0:(2^R - 1) for _ in 1:M]
        xranges = [0:(2^R - 1) for _ in 1:N]

        # Iterate over all combinations of input and output variables
        for x_vals in Iterators.product(xranges...), y_vals in Iterators.product(yranges...)
            x = collect(x_vals)  # Convert input tuple to a vector
            Axb = A * x .+ b  # Apply the affine transformation
            if boundary == Quantics.PeriodicBoundaryConditions()
                Axb = mod.(Axb, 2^R)
            elseif boundary == Quantics.OpenBoundaryConditions()
                Axb = map(x -> 0 <= x <= 2^R - 1 ? x : nothing, Axb)
            end
            ref = Axb == collect(y_vals)  # Compare to the reference output

            # Calculate indices for input and output
            iy = 1 + sum(y_vals[i] * (2^R)^(i - 1) for i in 1:M)
            ix = 1 + sum(x_vals[i] * (2^R)^(i - 1) for i in 1:N)

            @test T[iy, ix] == ref  # Verify the transformation result
        end
    end

    testtests = Dict(
        (1, 1) => [(reshape([1], 1, 1), [1])],
        (1, 2) => [([1 0], [0]), ([2 -1], [1])],
        (2, 1) => [([1; 0], [0, 0]), ([2; -1], [1, -1])],
        (2, 2) => [([1 0; 1 1], [0; 1]), ([2 0; 4 1], [100; -1])]
    )

    vars = ("x", "y", "z")
    insite = [Index(2; tags="$v$l") for l in 1:10, v in vars]
    outsite = [Index(2; tags="$v$l")' for l in 1:10, v in vars]
    boundaries = Quantics.OpenBoundaryConditions(), Quantics.PeriodicBoundaryConditions()

    @testset "full" begin
        A = [1 0; 1 1]
        b = [0; 0]

        T = Quantics.affine_transform_matrix(4, A, b)
        @test T' * T == I
        @test T[219, 59] == 1

        b = [4; 1]
        T = Quantics.affine_transform_matrix(4, A, b)
        @test T * T' == I
    end

    @testset "full R=$R, boundary=$(boundary), M=$M, N=$N" for R in [1, 2],
        boundary in boundaries, M in [1, 2], N in [1, 2]

        for (A, b) in testtests[(M, N)]
            A_ = reshape(A, M, N)
            b_ = reshape(b, M)
            T = Quantics.affine_transform_matrix(R, A_, b_, boundary)
            test_affine_transform_matrix_multi_variables(R, A_, b_, T, boundary)
        end
    end
    @testset "compare_simple" begin
        A = [1 0; 1 1]
        b = [0; 0]
        R = 3

        T = Quantics.affine_transform_matrix(R, A, b)
        mpo = Quantics.affine_transform_mpo(
            outsite[1:R, 1:2], insite[1:R, 1:2], A, b)
        Trec = Quantics.affine_mpo_to_matrix(
            outsite[1:R, 1:2], insite[1:R, 1:2], mpo)
        @test T == Trec
    end

    @testset "compare_hard" begin
        A = [1 0 1; 1 2 -1; 0 1 1]
        b = [11; 23; -15]
        R = 4

        T = Quantics.affine_transform_matrix(R, A, b)
        M, N = size(A)
        mpo = Quantics.affine_transform_mpo(
            outsite[1:R, 1:M], insite[1:R, 1:N], A, b)
        Trec = Quantics.affine_mpo_to_matrix(
            outsite[1:R, 1:M], insite[1:R, 1:N], mpo)
        @test T == Trec
    end

    @testset "compare_rect" begin
        A = [1 0 1; 1 2 0]
        b = [11; -3]
        R = 4

        T = Quantics.affine_transform_matrix(R, A, b)
        M, N = size(A)
        mpo = Quantics.affine_transform_mpo(
            outsite[1:R, 1:M], insite[1:R, 1:N], A, b)
        Trec = Quantics.affine_mpo_to_matrix(
            outsite[1:R, 1:M], insite[1:R, 1:N], mpo)
        @test T == Trec
    end

    @testset "compare_denom_odd" begin
        A = reshape([1 // 3], 1, 1)
        b = [0]

        for R in [1, 3, 6]
            for bc in boundaries
                T = Quantics.affine_transform_matrix(R, A, b, bc)
                M, N = size(A)
                mpo = Quantics.affine_transform_mpo(
                    outsite[1:R, 1:M], insite[1:R, 1:N], A, b, bc)
                Trec = Quantics.affine_mpo_to_matrix(
                    outsite[1:R, 1:M], insite[1:R, 1:N], mpo)
                @test T == Trec
            end
        end
    end

    @testset "compare_denom_even" begin
        A = reshape([1 // 2], 1, 1)

        for b in [[3], [5], [-3], [-5]]
            for R in [3, 5]
                #for bc in boundaries
                for bc in [Quantics.PeriodicBoundaryConditions()]
                    T = Quantics.affine_transform_matrix(R, A, b, bc)
                    M, N = size(A)
                    mpo = Quantics.affine_transform_mpo(
                        outsite[1:R, 1:M], insite[1:R, 1:N], A, b, bc)
                    Trec = Quantics.affine_mpo_to_matrix(
                        outsite[1:R, 1:M], insite[1:R, 1:N], mpo)
                    @test T == Trec
                end
            end
        end
    end

    @testset "compare_light_cone" begin
        A = 1 // 2 * [1 1; 1 -1]
        b = [2; 3]

        for R in [3, 4]
            for bc in boundaries
                T = Quantics.affine_transform_matrix(R, A, b, bc)
                M, N = size(A)
                mpo = Quantics.affine_transform_mpo(
                    outsite[1:R, 1:M], insite[1:R, 1:N], A, b, bc)
                Trec = Quantics.affine_mpo_to_matrix(
                    outsite[1:R, 1:M], insite[1:R, 1:N], mpo)
                @test T == Trec
            end
        end
    end

    @testset "degenerate_rational (issue #46)" begin
        # This transformation has no valid integer mappings because
        # A_int has all even entries while b_int[3] is odd, so
        # A_int*x + b_int can never be divisible by the even denominator s=42.
        A = [1 0 0; -1//7 6//7 6//7; -10//21 6//7 -1//7]
        b = [0, 27 // 7, 61 // 14]
        R = 3

        M, N = size(A)
        bc = Quantics.OpenBoundaryConditions()

        # The full matrix should be all zeros (no valid mappings)
        T = Quantics.affine_transform_matrix(R, A, b, bc)
        @test nnz(T) == 0

        # The MPO construction should not crash (was DivideError before fix)
        # and should emit a warning about no valid mappings.
        mpo = @test_warn "no valid integer mappings" Quantics.affine_transform_mpo(
            outsite[1:R, 1:M], insite[1:R, 1:N], A, b, bc)
        Trec = Quantics.affine_mpo_to_matrix(
            outsite[1:R, 1:M], insite[1:R, 1:N], mpo)
        @test T == Trec
    end
end
