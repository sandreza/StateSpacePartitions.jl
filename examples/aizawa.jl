using StateSpacePartitions, ProgressBars, Random, GLMakie, SparseArrays
include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.1
iterations = 10^3

timeseries = zeros(3, iterations)
timeseries[:, 1] .= [0.1, 0.0, 0.0] # a point on the attractor
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(aizawa, timeseries[:, i-1], dt)
    timeseries[:, i] .= step.xⁿ⁺¹
end

minimum_probability = 0.03
state_space_partitions = StateSpacePartition(timeseries; method = Tree(false, minimum_probability))

length(union(state_space_partitions.partitions)) == maximum(union(state_space_partitions.partitions))
visualize(timeseries, state_space_partitions)