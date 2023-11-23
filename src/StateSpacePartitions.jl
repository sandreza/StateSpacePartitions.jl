module StateSpacePartitions

using ProgressBars
using Reexport, PrecompileTools

include("Trees/Trees.jl")

export StateSpacePartition

import StateSpacePartitions.Trees: unstructured_tree, UnstructuredTree
struct StateSpacePartition{E, P}
    embedding::E 
    partitions::P 
end

function StateSpacePartition(timeseries; pmin = 0.01)
    @info "determine partitioning function "
    F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(timeseries, pmin)
    embedding = UnstructuredTree(P4, C, P3)
    me = Int64[]
    @info "computing markov embedding"
    for state in ProgressBar(eachcol(timeseries))
        push!(me, embedding(state))
    end
    return StateSpacePartition(embedding, me)
end

end # module StateSpacePartitions
