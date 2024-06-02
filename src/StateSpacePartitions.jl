module StateSpacePartitions

export StateSpacePartition

using ProgressBars
using Reexport, PrecompileTools

include("Trees/Trees.jl")

using .Trees

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

# Extensions
include("Extensions/markov_chain_hammer_extensions.jl")
include("Extensions/tree_extensions.jl")

@setup_workload begin
    using Random
    Random.seed!(1234)
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    @compile_workload begin
        trajectory = randn(3, 1000)
        ssp = StateSpacePartition(trajectory)
    end
end

end # module StateSpacePartitions
