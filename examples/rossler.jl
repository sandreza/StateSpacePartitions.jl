using StateSpacePartitions, ProgressBars, Random, GLMakie, SparseArrays
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

minimum_probability = 0.03
state_space_partitions = StateSpacePartition(timeseries; method = Tree(false, minimum_probability))

length(union(state_space_partitions.partitions)) == maximum(union(state_space_partitions.partitions))
visualize(timeseries, state_space_partitions)

##
function graph_from_PI(PI)
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    adj = spzeros(Int64, N, N)
    adj_mod = spzeros(Float64, N, N)
    for i in ProgressBar(eachindex(PI))
        ii = PI[i][1]
        jj = PI[i][2]
        modularity_value = PI[i][3]
        adj[ii, jj] += 1
        adj_mod[ii, jj] = modularity_value
    end 
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    node_labels = zeros(N)
    for i in eachindex(PI)
        node_labels[PI[i][2]] = PI[i][3]
    end
    
    return node_labels, adj, adj_mod, length(PI)
end
F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(timeseries, minimum_probability)
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