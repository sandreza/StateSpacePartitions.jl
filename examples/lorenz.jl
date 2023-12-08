using StateSpacePartitions, ProgressBars, Random, GLMakie 
include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.01 
iterations = 10^5

trajectory = zeros(3, iterations)
trajectory[:, 1] .= [14.0, 20.0, 27.0]
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(lorenz, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹
end

state_space_partitions = StateSpacePartition(trajectory)

Random.seed!(1234)
@info "determine partitioning function "
method = Tree()
embedding = determine_partition(trajectory, method)
partitions = zeros(Int64, size(trajectory)[2])
@info "computing partition trajectory"
for i in ProgressBar(eachindex(partitions))
    @inbounds partitions[i] = embedding(trajectory[:, i])
end

all(state_space_partitions.partitions .== partitions)

visualize(trajectory, state_space_partitions)