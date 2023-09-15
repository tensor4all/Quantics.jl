@testitem "quantics.jl" begin
using Test
using ITensors
using StaticArrays
import Quantics
import Quantics: QuanticsInd, QubitInd, index_to_fused_quantics
import Quantics: fused_quantics_to_qubit,
    qubit_to_fused_quantics, fused_quantics_to_index,
    index_to_qubit
import Quantics: qubit_to_index

function _to_ntuple(v::MVector{N,T}) where {N,T}
    return v.data
end
    @testset "QuanticsInd" begin
        Quantics.QuanticsInd{2}.([1, 2]) == Quantics.QuanticsInd(Val(2), [1, 2])
    end

    @testset "fused_quantics_to_qubit" begin
        @test fused_quantics_to_qubit(QuanticsInd{2}(1)) == Tuple(QubitInd.((1, 1)))
        @test fused_quantics_to_qubit(QuanticsInd{2}(2)) == Tuple(QubitInd.((2, 1)))
        @test fused_quantics_to_qubit(QuanticsInd{2}(3)) == Tuple(QubitInd.((1, 2)))
        @test fused_quantics_to_qubit(QuanticsInd{2}(4)) == Tuple(QubitInd.((2, 2)))
        @test fused_quantics_to_qubit(QuanticsInd{2}.([1, 2, 3, 4])) ==
              QubitInd.([1, 1, 2, 1, 1, 2, 2, 2])
    end

    @testset "index_to_qubit_1D" begin
        # 1D case
        R = 3
        @test index_to_qubit(1, R) == QubitInd.([1, 1, 1])
        @test index_to_qubit(2, R) == QubitInd.([1, 1, 2])
        @test index_to_qubit(3, R) == QubitInd.([1, 2, 1])
        @test index_to_qubit(4, R) == QubitInd.([1, 2, 2])
        @test index_to_qubit(5, R) == QubitInd.([2, 1, 1])
        @test index_to_qubit(6, R) == QubitInd.([2, 1, 2])
        @test index_to_qubit(7, R) == QubitInd.([2, 2, 1])
        @test index_to_qubit(8, R) == QubitInd.([2, 2, 2])
    end

    @testset "index_to_qubit_2D" begin
        D = 2
        R = 3 # Three bits along each axis
        for i in 1:(2^R), j in 1:(2^R)
            # index => fused quantics => index
            fused_quantics = index_to_fused_quantics((i, j), R)
            index = fused_quantics_to_index(fused_quantics)
            @test index == (i, j)

            # index => qubit => index
            qubit = index_to_qubit((i, j), R)
            @test qubit_to_index(Val(D), qubit, R) == (i, j)

            # fused quantics => qubit
            @test qubit == fused_quantics_to_qubit(fused_quantics)
            @test qubit_to_fused_quantics(Val(D), qubit) == fused_quantics
        end
    end

    @testset "fused_quantics_to_index" begin
        # quantics => qubit
        # 1        => (1, 1)
        # 4        => (2, 2)
        # 3        => (1, 2)
        #
        # index = (3, 4)
        D = 2 # Two-dimensional space
        R = 3 # Number of bits along each axis
        @test fused_quantics_to_qubit(QuanticsInd{D}.([1, 4, 3])) ==
              QubitInd.([1, 1, 2, 2, 1, 2])
        @test fused_quantics_to_index(QuanticsInd{D}.([1, 4, 3])) == (3, 4)
        @test index_to_fused_quantics((3, 4), R) == QuanticsInd{D}.([1, 4, 3])
    end
end

nothing
