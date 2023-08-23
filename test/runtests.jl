using Test

@testset "Quantics.jl" begin
include("util.jl")
include("tag.jl")
include("quanticsrepr.jl")
include("binaryop.jl")
include("fouriertransform.jl")
include("imaginarytime.jl")
include("mul.jl")
include("mps.jl")
include("transformer.jl")
include("grid.jl")
end

nothing
