using StateSpacePartitions, ProgressBars, Random, GLMakie 
include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.001 
iterations = 10^8

trajectory = zeros(3, iterations)
trajectory[:, 1] .= [14.0, 20.0, 27.0]
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(lorenz, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹ .+ sqrt(dt) * randn(3)
end

method = Tree(false, 0.001)
state_space_partitions = StateSpacePartition(trajectory; method = method)

length(union(state_space_partitions.partitions))
# visualize_koopman_mode(trajectory, state_space_partitions.partitions, colormap1 = :balance, mode = 4)