struct UnstructuredTree{L, C, CH}
    leafmap::L 
    centers::C 
    children::CH
end

UnstructuredTree() = UnstructuredTree([], [], [])

function (embedding::UnstructuredTree)(state)
    current_index = 1
    while length(embedding.centers[current_index]) > 1
        local_child = argmin([norm(state - center) for center in embedding.centers[current_index]])
        current_index = embedding.children[current_index][local_child]
    end
    return embedding.leafmap[current_index]
end

function split(trajectory, indices, n_min; numstates = 2)
    if length(indices) > n_min
        r0 = kmeans(view(trajectory, :, indices), numstates; max_iters=10^4)
        inds = [[i for (j, i) in enumerate(indices) if r0.assignments[j] == k] for k in 1:numstates]
        centers = [r0.centers[:, k] for k in 1:numstates]
        return inds, centers
    end
    return [[]], [[]]
end

# modification of code from Peter J. Schmid
function unstructured_tree(trajectory, p_min; threshold = 2)
    n = size(trajectory)[2]
    n_min = floor(Int, threshold * p_min * n)
    W, F, P1, edge_information = [collect(1:n)], [], [1], []
    H = []
    centers_list = Dict()
    parent_to_children = Dict()
    global_to_local = Dict()
    local_to_global = Dict()
    CC = Dict()
    leaf_index = 1
    global_index = 1
    while (length(W) > 0)
        w = popfirst!(W)
        p1 = popfirst!(P1)
        inds, centers = split(trajectory, w, n_min)
        centers_list[p1] =  centers
        if all([length(ind) > 0 for ind in inds])
            W = [inds..., W...]
            Ptmp = []
            [push!(Ptmp, global_index + i) for i in eachindex(inds)]
            P1 = [Ptmp..., P1...]
            [push!(edge_information, (p1, global_index + i, length(ind) / n)) for (i, ind) in enumerate(inds)]
            Ptmp2 = Int64[]
            [push!(Ptmp2, global_index + i) for (i, ind) in enumerate(inds)]
            [CC[global_index + i] = centers[i] for i in eachindex(inds)]
            parent_to_children[p1] = Ptmp2
            global_index += length(inds)
            push!(H, [inds...])
        else
            push!(F, w)
            push!(H, [[]])
            parent_to_children[p1] = NaN
            global_to_local[p1] = leaf_index
            local_to_global[leaf_index] = p1
            leaf_index += 1
        end
    end
    return F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global
end

function determine_partition(trajectory, tree_type::Tree{Val{false}, S}; override = false) where S
    if typeof(tree_type.arguments) <: NamedTuple
        if haskey(tree_type.arguments, :minimum_probability)
            minimum_probability = tree_type.arguments.minimum_probability 
        else
            @warn "no minimum probability specified, using 0.01"
            minimum_probability = 0.01
        end
    elseif typeof(tree_type.arguments) <: Number
        minimum_probability = tree_type.arguments
    else
        @warn "no minimum probability specified, using 0.01"
        minimum_probability = 0.01
    end
    Nmax = 100 * round(Int, 1/ minimum_probability)
    if (size(trajectory)[2] > Nmax) & !(override)
        @warn "trajectory too long, truncating to roughly $Nmax for determining embedding"
        skip = round(Int, size(trajectory)[2] / Nmax)
        trajectory = trajectory[:, 1:skip:end]
    end
    if (10/(size(trajectory)[2]) > minimum_probability) & !(override)
        @warn "minimum probabity too small, using 10x the reciprocal of the number of points"
        minimum_probability = 10 / size(trajectory)[2]
    end
    F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(trajectory, minimum_probability)
    embedding = UnstructuredTree(global_to_local, centers_list, parent_to_children)
    return embedding
end
