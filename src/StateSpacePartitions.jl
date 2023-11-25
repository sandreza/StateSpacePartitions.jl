module StateSpacePartitions

using ProgressBars
using Reexport, PrecompileTools

include("Trees/Trees.jl")

export StateSpacePartition

import StateSpacePartitions.Trees: unstructured_tree, UnstructuredTree, determine_partition, Tree
struct StateSpacePartition{E, P}
    embedding::E 
    partitions::P 
end

function StateSpacePartition(timeseries; method = Tree(), override = false)
    @info "determine partitioning function "
    embedding = determine_partition(timeseries, method; override = override)
    partitions = zeros(Int64, size(timeseries)[2])
    @info "computing partition timeseries"
    for i in ProgressBar(eachindex(partitions))
        @inbounds partitions[i] = embedding(timeseries[:, i])
    end
    return StateSpacePartition(embedding, partitions)
end

end # module StateSpacePartitions
