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

function StateSpacePartition(trajectory; method = Tree(), override = false)
    @info "determine partitioning function "
    embedding = determine_partition(trajectory, method; override = override)
    partitions = zeros(Int64, size(trajectory)[2])
    @info "computing partition trajectory"
    for i in ProgressBar(eachindex(partitions))
        @inbounds partitions[i] = embedding(trajectory[:, i])
    end
    return StateSpacePartition(embedding, partitions)
end

include("inverse_iteration.jl")
include("coarsen.jl")
include("markov_chain_hammer_extensions.jl")

end # module StateSpacePartitions
