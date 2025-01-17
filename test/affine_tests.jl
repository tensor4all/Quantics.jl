@testitem "affine" begin
    using Test
    using ITensors
    using Quantics
    using LinearAlgebra

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
        A = reshape([1//3], 1, 1)
        b = [0]

        for R in [2, 3, 6]
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
        A = reshape([1//2], 1, 1)
        b = [3]   # XXX b = 5 would not work :(

        for R in [3, 5]
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

    @testset "compare_light_cone" begin
        A = 1//2 * [1 1; 1 -1]
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

end
