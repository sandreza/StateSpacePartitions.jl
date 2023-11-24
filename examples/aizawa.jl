using StateSpacePartitions, ProgressBars, Random, GLMakie, SparseArrays
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

function aizawa(s; a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.25)
    x, y, z = s
    ẋ = (z - b) * x - d * y
    ẏ = d * x + (z - b) * y
    ż = c + a * z - z^3/3 - (x^2 + y^2) * (1 + e * z) + f * z * x^3
    return [ẋ, ẏ, ż]
end

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