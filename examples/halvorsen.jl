using StateSpacePartitions, ProgressBars, Random, GLMakie, SparseArrays

include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.1
iterations = 10^8

trajectory = zeros(3, iterations)
trajectory[:, 1] .= [ -3.5626767040251024, -4.862155361130287, 4.104823253572631] # a point on the attractor
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(halvorsen, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹
end

minimum_probability = 0.001
@time state_space_partitions = StateSpacePartition(trajectory; chunk_size = size(trajectory, 2) ÷ 50, method = Tree(false, minimum_probability))

visualize_koopman_mode(trajectory, partitions, colormap1 = :thermal, mode = 2)