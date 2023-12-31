using StateSpacePartitions, ProgressBars, Random, GLMakie, SparseArrays
include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.1
iterations = 10^4

trajectory = zeros(3, iterations)
trajectory[:, 1] .= [  0.17365637238010478, -0.07044917391587771, 0.3761221214080355] # a point on the attractor
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(piecewise, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹
end

minimum_probability = 0.001/4
state_space_partitions = StateSpacePartition(trajectory; method = Tree(false, minimum_probability))

visualize_koopman_mode(trajectory, state_space_partitions.partitions, colormap1 = :thermal, mode = 2)