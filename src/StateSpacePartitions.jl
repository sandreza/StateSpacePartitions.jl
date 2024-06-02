module StateSpacePartitions

export StateSpacePartition

using ProgressBars
using Reexport, PrecompileTools

include("Trees/Trees.jl")

using .Trees

# import StateSpacePartitions.Trees: unstructured_tree, UnstructuredTree, determine_partition, Tree
struct StateSpacePartition{E, P}
    embedding::E 
    partitions::P 
end

function StateSpacePartition(trajectory; 
                             method = Tree(), 
                             cells = nothing,
                             override = false)

    @info "determine partitioning function "
    if  isnothing(cells)
    elseif typeof(cells) == Int
        @assert cells > 0
        method = Tree(; structured = false, arguments = (; minimum_probability = 1/cells))
    else
        @error "cells must be an integer"
    end

    embedding = determine_partition(trajectory, method; override = override)
    partitions = zeros(Int64, size(trajectory)[2])

    @info "computing partition trajectory"
    for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
        partitions[i] = embedding(state)
    end

    return StateSpacePartition(embedding, partitions)
end

# Utility Functions

include("inverse_iteration.jl")
include("markov_chain_hammer_extensions.jl")
include("tree_extensions.jl")

end # module StateSpacePartitions
