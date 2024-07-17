#__precompile__(false) 

module Quantics

#@everywhere begin
#using Pkg
#Pkg.activate(".")
#Pkg.instantiate()
#end

using ITensors
import ITensors
import ITensors.NDTensors: Tensor, BlockSparseTensor, blockview

import SparseIR: Fermionic, Bosonic, Statistics
import LinearAlgebra: I
using StaticArrays

import FastMPOContractions
import ITensorTDVP

using EllipsisNotation

function __init__()
end

include("util.jl")
include("sum.jl")
include("tag.jl")
include("binaryop.jl")
include("mul.jl")
include("mps.jl")
include("fouriertransform.jl")
include("imaginarytime.jl")
include("transformer.jl")

end
