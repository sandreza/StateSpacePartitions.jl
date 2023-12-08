using StateSpacePartitions, ProgressBars, Random, GLMakie, SparseArrays
include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.1
iterations = 10^3

trajectory = zeros(3, iterations)
trajectory[:, 1] .= [ -0.5705898323406506, -0.14829855508752873, 1.170754662354835] # a point on the attractor
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(sprott_case_a, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹
end

minimum_probability = 0.03
state_space_partitions = StateSpacePartition(trajectory; method = Tree(false, minimum_probability))

visualize(trajectory, state_space_partitions)