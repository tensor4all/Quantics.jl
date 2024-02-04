@testitem begin
    using Aqua
    import Quantics

    @testset "Aqua" begin
        Aqua.test_stale_deps(Quantics)
    end
end