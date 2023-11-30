# goals: rename things into something more intuitive
# P3: parent_to_children
# P4: global index -> local index
# G: Graph, has parent nodes, G.badjlist yields parent node. e.g. 
# G.badjlist[1] = [] means no parent, G.badjlist[2] = [1] means parent is 1
# G.fadjlist yields children nodes, e.g. G.fadjlist[1] = [2, 3] means children are 2 and 3
# C is the centers
# PI -> Edge has the (parent_index, child_index, probability)  it is ordered by the child index 
# PI should be of type Vector{Tuple{Int64, Int64, Float64}}
## 
function leaf_global_to_local_indices(graph_edges)
    parents = [graph_edges[i][1] for i in eachindex(graph_edges)]
    children = [graph_edges[i][2] for i in eachindex(graph_edges)]
    childless = setdiff(children, parents)
    global_to_local = Dict()
    local_to_global = Dict()
    j = 0
    for i in eachindex(childless) 
        j += 1 
        global_to_local[childless[i]] = j
        local_to_global[j] = childless[i]
    end
    return global_to_local, local_to_global
end

# This needs improvement, still a bit too slow
function unstructured_coarsen_edges(graph_edges, probability_minimum, parent_to_children, G, global_to_local)
    tic = time()
    toc = time()
    # @info "part 1: $(toc - tic)"
    parents = [graph_edges[i][1] for i in eachindex(graph_edges)]
    children = [graph_edges[i][2] for i in eachindex(graph_edges)]
    probabilities = [graph_edges[i][3] for i in eachindex(graph_edges)]
    parent_probabilities = Dict() 
    for parent in union(parents)
        parent_probabilities[parent] = 0
    end
    toc = time()
    # @info "part 2: $(toc - tic)"
    for i in eachindex(parents)
        parent_probabilities[parents[i]] += probabilities[i]
    end
    toc = time()
    # @info "part 3: $(toc - tic)"
    delete_children_list = Int64[]
    childless_list = Int64[]
    for parent in union(parents)
        if parent_probabilities[parent] < probability_minimum
            push!(childless_list, parent)
            #=
            if all(isnan.(parent_to_children[parent]))
                println("blame ", parent)
                break 
            end
            =#
            push!(delete_children_list, parent_to_children[parent]...)
        end
    end
    toc = time()
    # @info "part 4: $(toc - tic)"
    new_parent_to_children = copy(parent_to_children)
    for child in delete_children_list
        delete!(new_parent_to_children, child)
    end
    toc = time()
    # @info "part 5: $(toc - tic)"
    for new_childless in childless_list
        new_parent_to_children[new_childless] = NaN
    end
    # The code below was generated
    toc = time()
    # @info "part 6: $(toc - tic)"
    new_graph_edges = Tuple{Int64, Int64, Float64}[]
    for i in eachindex(graph_edges)
        if !(graph_edges[i][1] in childless_list)
            push!(new_graph_edges, graph_edges[i])
        end
    end
    # The code above was generate
    # global_to_local, local_to_global = leaf_global_to_local_indices(graph_edges)
    toc = time()
    # @info "part 7: $(toc - tic)"
    local_to_global = Dict(value => key for (key, value) in global_to_local)
    new_global_to_local, new_local_to_global = leaf_global_to_local_indices(new_graph_edges)
    toc = time()
    # @info "part 8: $(toc - tic)"
    global_to_global = Dict()
    #=
    # this was a bad idea
    for key in ProgressBar(keys(global_to_local))
        if key in delete_children_list
            delete_children_list = setdiff(delete_children_list, key)
            # delete_children_list = setdiff!(delete_children_list, key)
            for ancestor_key in keys(new_global_to_local)
                if isancestor(ancestor_key, key, G)
                    global_to_global[key] = ancestor_key
                end 
            end 
        else
            global_to_global[key] = key
        end
    end
    =#
    for key in delete_children_list
        for ancestor_key in keys(new_global_to_local)
            if isancestor(ancestor_key, key, G)
                global_to_global[key] = ancestor_key
                break
            end 
        end 
    end
    other_keys = setdiff(keys(global_to_local), delete_children_list)
    for key in other_keys
        global_to_global[key] = key
    end
    toc = time()
    # @info "part 9: $(toc - tic)"
    local_to_local = Dict()
    for key in keys(local_to_global)
        local_to_local[key] = new_global_to_local[global_to_global[local_to_global[key]]]
    end
    return local_to_local
end

# can probably implemented recursively
function isancestor(ancestor, child, G)
    for i in eachindex(G.badjlist)
        if ancestor == G.badjlist[child][1]
            return true
        else
            child = G.badjlist[child][1]
        end
        if isempty(G.badjlist[child])
            return false 
        end
    end
    return nothing
end

#=
dt = 0.01 
iterations = 10^5

timeseries = zeros(3, iterations)
timeseries[:, 1] .= [14.0, 20.0, 27.0]
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(lorenz, timeseries[:, i-1], dt)
    timeseries[:, i] .= step.xⁿ⁺¹
end

p_min = 0.01

##
F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(timeseries, p_min; threshold = 2)
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI);
G = SimpleDiGraph(adj)

p_min = 0.01
F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(timeseries, p_min; threshold = 2)
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI);
G = SimpleDiGraph(adj)
graph_edges = PI 
probability_minimum = 0.1
parent_to_children = P3 
global_to_local = P4
local_to_local = unstructured_coarsen_edges(graph_edges, probability_minimum, parent_to_children, G, global_to_local)

# parent_to_children = copy(P3)

# Does not correspond to P4
##
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI);
G = SimpleDiGraph(adj)
nn = maximum([PI[i][2] for i in eachindex(PI)]);
# node_labels_original = copy(node_labels) 
node_labels = collect(1:nn)
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
=#