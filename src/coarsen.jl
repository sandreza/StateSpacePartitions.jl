# Unstructured 
export coarsen, unstructured_coarsen_edges

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
    parents = [graph_edges[i][1] for i in eachindex(graph_edges)]
    children = [graph_edges[i][2] for i in eachindex(graph_edges)]
    probabilities = [graph_edges[i][3] for i in eachindex(graph_edges)]
    parent_probabilities = Dict() 
    for parent in union(parents)
        parent_probabilities[parent] = 0
    end
    for i in eachindex(parents)
        parent_probabilities[parents[i]] += probabilities[i]
    end
    delete_children_list = Int64[]
    childless_list = Int64[]
    for parent in union(parents)
        if parent_probabilities[parent] < probability_minimum
            push!(childless_list, parent)
            push!(delete_children_list, parent_to_children[parent]...)
        end
    end
    new_parent_to_children = copy(parent_to_children)
    for child in delete_children_list
        delete!(new_parent_to_children, child)
    end
    for new_childless in childless_list
        new_parent_to_children[new_childless] = NaN
    end
    # The code below was generated
    new_graph_edges = Tuple{Int64, Int64, Float64}[]
    for i in eachindex(graph_edges)
        if !(graph_edges[i][1] in childless_list)
            push!(new_graph_edges, graph_edges[i])
        end
    end
    # The code above was generate
    local_to_global = Dict(value => key for (key, value) in global_to_local)
    new_global_to_local, new_local_to_global = leaf_global_to_local_indices(new_graph_edges)
    global_to_global = Dict()
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

# Binary Tree 
function coarsen(ssp::StateSpacePartition{S, A}, levels) where {S <: BinaryTree, A} 
    partitions = ssp.partitions
    return (partitions .- 1) .÷ 2^levels .+ 1
end