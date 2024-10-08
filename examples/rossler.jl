using StateSpacePartitions, ProgressBars, Random, GLMakie, SparseArrays
include("chaotic_systems.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
Random.seed!(1234)

dt = 0.2
iterations = 10^4

trajectory = zeros(3, iterations)
trajectory[:, 1] .= [ -8.605439656793074, -1.6791366418368037, 0.013910051541844323] # a point on the attractor
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(rossler, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹
end

levels = 8
minimum_probability = 0.75/2^(levels)
tree_type = Tree(false, minimum_probability)
state_space_partitions = StateSpacePartition(trajectory; method = tree_type)

length(union(state_space_partitions.partitions)) == maximum(union(state_space_partitions.partitions))
visualize(trajectory, state_space_partitions)
##
tree_type = Tree(true, levels)
state_space_partitions_2 = StateSpacePartition(trajectory; method = tree_type)
##
fig = Figure() 
ax = LScene(fig[1,1]; show_axis=false)
scatter!(ax, trajectory, color=state_space_partitions.partitions, colormap=:glasbey_hv_n256, markersize=10)
ax = LScene(fig[1,2]; show_axis=false)
scatter!(ax, trajectory, color=state_space_partitions_2.partitions, colormap=:glasbey_hv_n256, markersize=10)
display(fig)
##
F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(trajectory, minimum_probability)
##
using GraphMakie, NetworkLayout, Graphs, Printf
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI);
nn = maximum([PI[i][2] for i in eachindex(PI)]);
fig = Figure(resolution=(2 * 800, 800))
layout = Buchheim()
colormap = :glasbey_hv_n256
set_theme!(backgroundcolor=:white)

ax11 = Axis(fig[1, 1])
G = SimpleDiGraph(adj)
transparancy = 0.4 * adj_mod.nzval[:] / adj_mod.nzval[1] .+ 0.1
nlabels_fontsize = 35
edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers]
nlabels = [@sprintf("%2.2f", node_labels[i]) for i in 1:nv(G)]
graphplot!(ax11, G, layout=layout, nlabels=nlabels, node_size=100,
    node_color=(:orange, 0.9), edge_color=edge_color, edge_width=5,
    arrow_size=45, nlabels_align=(:center, :center),
    nlabels_textsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
# cc = cameracontrols(ax11.scene)
hidedecorations!(ax11)
hidespines!(ax11);
display(fig)