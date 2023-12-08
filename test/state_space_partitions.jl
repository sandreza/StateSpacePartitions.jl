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
