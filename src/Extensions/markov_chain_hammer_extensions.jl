holding_times(state_space_partitions::StateSpacePartition; dt =dt ) = holding_times(state_space_partitions.partitions, maximum(state_space_partitions.partitions), dt=dt)
sparse_perron_frobenius(state_space_partitions::StateSpacePartition; step = 1) = sparse_perron_frobenius(state_space_partitions.partitions; step = step)
sparse_generator(state_space_partitions::StateSpacePartition; dt = 1) = sparse_generator(state_space_partitions.partitions; dt=dt)

function steady_state(state_space_partitions::StateSpacePartition)
    partitions = state_space_partitions.partitions
    p = zeros(maximum(partitions))
    for i in eachindex(partitions)
        p[partitions[i]] += 1
    end
    return p ./ sum(p)
end