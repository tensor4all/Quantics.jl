@testitem "mps.jl" begin
    using Test
    import Quantics
    using ITensors
    @testset "onemps" begin
        nbit = 3
        sites = siteinds("Qubit", nbit)
        M = Quantics.onemps(Float64, sites)
        @test vec(Array(reduce(*, M), sites)) ≈ ones(2^nbit)
    end

    @testset "expqtt" begin
        R = 10
        sites = siteinds("Qubit", 10)
        f = Quantics.expqtt(sites, -1.0)
        f_values = vec(Array(reduce(*, f), reverse(sites)))
        xs = collect(LinRange(0, 1, 2^R + 1)[1:(end - 1)])
        f_values_ref = (x -> exp(-x)).(xs)

        @test f_values ≈ f_values_ref
    end
end
