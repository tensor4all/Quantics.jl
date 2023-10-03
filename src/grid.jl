
abstract type Grid end

"""
Convert a grid index to the corresponding coordinate in the original coordinate system
"""
function grididx_to_origcoord(g::Grid, index::NTuple{N,Int}) where {N}
    all(1 .<= index .<= 2^g.R) || error("1 <= {index} <= 2^g.R")
    return (index .- 1) .* grid_step(g) .+ grid_min(g)
end

# Backward compatibility
originalcoordinate(g::Grid, index::NTuple{N,Int}) where {N} = grididx_to_origcoord(g, index)

"""
Convert an grid index to quantices indices
"""
function grididx_to_quantics(g::Grid, grididx::NTuple{N,Int}) where {N}
    index_to_fused_quantics(grididx, g.R)
end

# Backward compatibility
to_quantics(g::Grid, grididx::NTuple{N,Int}) where {N} = grididx_to_quantics(g, grididx)

"""
Convert quantics indices to the original coordinate system
"""
function quantics_to_origcoord(g::Grid, index::AbstractVector{QuanticsInd{N}}) where {N}
    return grididx_to_origcoord(g, fused_quantics_to_index(index))
end

# Backward compatibility
function originalcoordinate(g::Grid, index::AbstractVector{QuanticsInd{N}}) where {N}
    quantics_to_origcoord(g, index)
end

@doc raw"""
The InherentDiscreteGrid struct represents a grid for inherently discrete data.
The grid contains values at specific, 
equally spaced points, but these values do not represent discretized versions 
of continuous data. Instead, they represent individual data points that are 
inherently discrete.
The linear size of the mesh is ``2^R``.
"""
struct InherentDiscreteGrid{N} <: Grid
    R::Int
    origin::NTuple{N,Int}
end

grid_min(grid::InherentDiscreteGrid) = grid.origin
grid_step(grid::InherentDiscreteGrid{N}) where {N} = ntuple(i -> 1, N)

"""
Create a grid for inherently discrete data
"""
function InherentDiscreteGrid{N}(R::Int) where {N}
    return InherentDiscreteGrid{N}(R, ntuple(i -> 1, N))
end

"""
Convert a coordinate in the original coordinate system to the corresponding grid index
"""
function origcoord_to_grididx(g::InherentDiscreteGrid, coordinate::NTuple{N,Int}) where {N}
    return coordinate .- grid_min(g) .+ 1
end

# Backward compatibility
function gridpoint(g::InherentDiscreteGrid, coordinate::NTuple{N,Int}) where {N}
    origcoord_to_grididx(g, coordinate)
end

@doc raw"""
The DiscretizedGrid struct represents a grid for discretized continuous data.
This is used for data that is originally continuous,
but has been discretized for computational purposes.
The grid contains values at specific, equally spaced points, which represent discrete 
approximations of the original continuous data. 
"""
struct DiscretizedGrid{N} <: Grid
    R::Int
    grid_min::NTuple{N,Float64}
    grid_max::NTuple{N,Float64}
end

grid_min(g::DiscretizedGrid) = g.grid_min
grid_max(g::DiscretizedGrid) = g.grid_max
grid_step(g::DiscretizedGrid) = (g.grid_max .- g.grid_min) ./ (2^g.R)

"""
Create a discrete grid for continuous data
"""
function DiscretizedGrid{N}(R::Int) where {N}
    return DiscretizedGrid{N}(R, ntuple(i -> 0.0, N), ntuple(i -> 1.0, N))
end

"""
Convert a coordinate in the original coordinate system to the corresponding grid index
"""
function origcoord_to_grididx(g::DiscretizedGrid, coordinate::NTuple{N,Float64}) where {N}
    all(grid_min(g) .<= coordinate .< grid_max(g)) ||
        error("Bound Error: $(coordinate), min=$(grid_min(g)), max=$(grid_max(g))")
    return ((coordinate .- grid_min(g)) ./ grid_step(g) .+ 1) .|> floor .|> Int
end

function origcoord_to_grididx(g::DiscretizedGrid{1}, coordinate::Float64)
    origcoord_to_grididx(g, (coordinate,))[1]
end

# Backward compatibility
function gridpoint(g::DiscretizedGrid, coordinate::NTuple{N,Float64}) where {N}
    origcoord_to_grididx(g, coordinate)
end
gridpoint(g::DiscretizedGrid{1}, coordinate::Float64) = gridpoint(g, (coordinate,))[1]
