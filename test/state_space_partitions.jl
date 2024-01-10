using StateSpacePartitions, Random 

Random.seed!(12345)

@testset "State Space Partitions" begin
    states = 3
    trajectory = 10^2
    trajectory = randn(states, trajectory)
    state_space_partitions = StateSpacePartition(trajectory)
    @test maximum(state_space_partitions.partitions) ≤ 100
    method = Tree(true, 4)
    state_space_partitions = StateSpacePartition(trajectory; method)
    @test maximum(state_space_partitions.partitions) == 2^4
    method = Tree(false, 0.1)
    state_space_partitions = StateSpacePartition(trajectory; method)
    @test maximum(state_space_partitions.partitions) ≤ 10
    method = Tree(structured = false, arguments = (; minimum_probability = 0.01))
    @test maximum(state_space_partitions.partitions) ≤ 100
    method = Tree(structured = true, arguments = 4)
    state_space_partitions = StateSpacePartition(trajectory; method)
    @test maximum(state_space_partitions.partitions) == 16
end

@testset "Full trajectory embedding" begin
    states = 3
    trajectory = 10^2
    trajectory = randn(states, trajectory)

    Random.seed!(12345)

    state_space_partitions = StateSpacePartition(trajectory)
    full_partitions = state_space_partitions.partitions

    embedding  = state_space_partitions.embedding
    partitions = zeros(Int64, size(trajectory, 2))
    
    for i in 1:size(trajectory, 2)
        partitions[i] = embedding(trajectory[:, i])
    end

    @test partitions == full_partitions
end