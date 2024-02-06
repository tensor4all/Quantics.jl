@testitem "Code quality test with Aqua.jl" begin
    using Aqua
    import Quantics

    @testset "Aqua" begin
        Aqua.test_all(Quantics, ambiguities = false, deps_compat = false)
    end
end