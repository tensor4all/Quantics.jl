@testitem "Code linting with JET.jl" begin
    using JET
    import Quantics

    if VERSION ≥ v"1.9"
        @testset "JET" begin
            JET.test_package(Quantics; target_defined_modules=true)
        end
    end
end