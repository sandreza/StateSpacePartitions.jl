using StateSpacePartitions, ProgressBars, Random, GLMakie 
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

function rossler(s; a = 0.2, b = 0.2, c = 5.7)
    x, y, z = s
    ẋ = -y - z
    ẏ = x + a * y
    ż = b + z * (x - c)
    return [ẋ, ẏ, ż]
end

dt = 0.1
iterations = 10^4

timeseries = zeros(3, iterations)
timeseries[:, 1] .= [ -8.605439656793074, -1.6791366418368037, 0.013910051541844323] # a point on the attractor
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(rossler, timeseries[:, i-1], dt)
    timeseries[:, i] .= step.xⁿ⁺¹
end

state_space_partitions = StateSpacePartition(timeseries; method = Tree(false, 0.1))

length(union(state_space_partitions.partitions)) == maximum(union(state_space_partitions.partitions))
visualize(timeseries, state_space_partitions)