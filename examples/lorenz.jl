using StateSpacePartitions, ProgressBars, Random 
include("timestepping_utils.jl")
Random.seed!(1234)

function lorenz!(ṡ, s)
    ṡ[1] = 10.0 * (s[2] - s[1])
    ṡ[2] = s[1] * (28.0 - s[3]) - s[2]
    ṡ[3] = s[1] * s[2] - (8 / 3) * s[3]
    return ṡ
end

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