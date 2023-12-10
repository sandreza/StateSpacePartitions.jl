using StateSpacePartitions, ProgressBars, Random, GLMakie 
include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.01 
iterations = 10^6

trajectory = zeros(3, iterations)
trajectory[:, 1] .= [14.0, 20.0, 27.0]
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(lorenz, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹
end

state_space_partitions = StateSpacePartition(trajectory; method = Tree(false, 0.001))

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

##
using MarkovChainHammer, Graphs, NetworkLayout, GraphMakie, LinearAlgebra
P = perron_frobenius(state_space_partitions.partitions)
P2 = perron_frobenius(reverse(state_space_partitions.partitions))
P = (P2 + P)/2
P = P - Diagonal(P)
P = P ./ sum(P, dims = 1)
g = DiGraph(P)
P2 = P2 - Diagonal(P2)
P2 = P2 ./ sum(P2, dims = 1)
g2 = DiGraph(P2)
##
fig = Figure() 
layout = Spectral()
ax = LScene(fig[1, 1])
graphplot!(ax, g, layout=layout, node_size=0.0, edge_width=1.0)
ax = LScene(fig[1, 2])
graphplot!(ax, g2, layout=layout, node_size=0.0, edge_width=1.0)
display(fig)
