using StateSpacePartitions, ProgressBars, Random, GLMakie 
include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.05
iterations = 10^4
ϵ = 0.05

trajectory = zeros(3, iterations)
trajectory[:, 1] .= [0.21690008444818204, -0.26299612444316517, -0.2424012650991593]
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(newton_leipnick, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹ .+ ϵ * randn(3) * sqrt(dt)
end

state_space_partitions = StateSpacePartition(trajectory; method = Tree(; structured = false, arguments = 0.1))

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