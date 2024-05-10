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

method = Tree(false, 0.001)
state_space_partitions = StateSpacePartition(trajectory; method = method)

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
Qᶠ = generator(state_space_partitions.partitions)
Qᵇ = generator(reverse(state_space_partitions.partitions))
[Qᶠ[i, i] = 0 for i in 1:size(Qᶠ)[1]]
[Qᵇ[i, i] = 0 for i in 1:size(Qᵇ)[1]]
Qʳ = (Qᶠ + Qᵇ)/2
Qⁱ = (Qᶠ - Qᵇ)/2
gᶠ = DiGraph(Qᶠ)
gᵇ = DiGraph(Qᵇ)
gʳ = DiGraph(Qʳ)
gⁱ = DiGraph(Qⁱ)
##
set_theme!(backgroundcolor = :white)
fig = Figure() 
layout = Spectral(dim = 3)
ax = LScene(fig[1, 1]; show_axis = false)
graphplot!(ax, gᶠ, layout=layout, node_size=0.0, edge_width=1.0)
ax = LScene(fig[1, 2]; show_axis = false)
graphplot!(ax, gᵇ, layout=layout, node_size=0.0, edge_width=1.0)
ax = LScene(fig[2, 1]; show_axis = false)
graphplot!(ax, gʳ, layout=layout, node_size=0.0, edge_width=1.0)
ax = LScene(fig[2, 2]; show_axis = false)
graphplot!(ax, gⁱ, layout=layout, node_size=0.0, edge_width=1.0)
display(fig)
