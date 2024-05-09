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
                             override = false)

    @info "determine partitioning function "
    embedding = determine_partition(trajectory, method; override = override, architecture)
    partitions = zeros(Int64, size(trajectory)[2])
    # partitions = ChunkedArray(partitions, architecture; chunk_size)
    # chunked_trajectory = ChunkedArray(trajectory, architecture; chunk_size)
    # embedding(partitions, chunked_trajectory)

    @info "computing partition trajectory"
    for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
        partitions[i] = embedding(state)
    end

    return StateSpacePartition(embedding, partitions)
end

include("inverse_iteration.jl")
include("coarsen.jl")
include("markov_chain_hammer_extensions.jl")

end # module StateSpacePartitions
