using StateSpacePartitions, ProgressBars, Random, GLMakie 
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

function lorenz(s)
    x, y, z = s
    ẋ = 10.0 * (y - x)
    ẏ = x * (28.0 - z) - y
    ż = x * y - (8 / 3) * z
    return [ẋ, ẏ, ż]
end

dt = 0.1 
iterations = 10^5

timeseries = zeros(3, iterations)
timeseries[:, 1] .= [14.0, 20.0, 27.0]
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(lorenz, timeseries[:, i-1], dt)
    timeseries[:, i] .= step.xⁿ⁺¹
end

state_space_partitions = StateSpacePartition(timeseries)

Random.seed!(1234)
@info "determine partitioning function "
method = Tree()
embedding = determine_partition(timeseries, method)
partitions = zeros(Int64, size(timeseries)[2])
@info "computing partition timeseries"
for i in ProgressBar(eachindex(partitions))
    @inbounds partitions[i] = embedding(timeseries[:, i])
end

all(state_space_partitions.partitions .== partitions)

visualize(timeseries, state_space_partitions)