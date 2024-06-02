module StateSpacePartitions

export StateSpacePartition

using ProgressBars
using Reexport, PrecompileTools

include("Architectures.jl")
include("Trees/Trees.jl")

using .Architectures
using .Trees

# import StateSpacePartitions.Trees: unstructured_tree, UnstructuredTree, determine_partition, Tree
struct StateSpacePartition{E, P}
    embedding::E 
    partitions::P 
end

function StateSpacePartition(trajectory; 
                             architecture = CPU(),
                             method = Tree(), 
                             chunk_size = size(trajectory)[2],
                             cells = nothing,
                             chunked = false,
                             override = false)

    @info "determine partitioning function "
    if  isnothing(cells)
    elseif typeof(cells) == Int
        @assert cells > 0
        method = Tree(; structured = false, arguments = (; minimum_probability = 1/cells))
    else
        @error "cells must be an integer"
    end

    embedding = determine_partition(trajectory, method; override = override, architecture)
    partitions = zeros(Int64, size(trajectory)[2])

    if chunked
        @info "computing (chunked) partition trajectory"
        chunked_partitions = ChunkedArray(partitions, architecture; chunk_size)
        chunked_trajectory = ChunkedArray(trajectory, architecture; chunk_size)
        embedding(chunked_partitions, chunked_trajectory)
        return StateSpacePartition(embedding, chunked_partitions)
    else
        @info "computing partition trajectory"
        for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
            partitions[i] = embedding(state)
        end
        return StateSpacePartition(embedding, partitions)
    end
    return nothing
end

include("inverse_iteration.jl")
include("coarsen.jl")
include("markov_chain_hammer_extensions.jl")

end # module StateSpacePartitions
