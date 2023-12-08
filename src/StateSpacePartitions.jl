module StateSpacePartitions

using ProgressBars
using Reexport, PrecompileTools

include("Trees/Trees.jl")

export StateSpacePartition

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

end # module StateSpacePartitions
