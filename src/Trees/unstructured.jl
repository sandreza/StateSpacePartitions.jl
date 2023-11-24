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

function split(timeseries, indices, n_min; numstates = 2)
    if length(indices) > n_min
        r0 = kmeans(view(timeseries, :, indices), numstates; max_iters=10^4)
        inds = [[i for (j, i) in enumerate(indices) if r0.assignments[j] == k] for k in 1:numstates]
        centers = [r0.centers[:, k] for k in 1:numstates]
        return inds, centers
    end
    return [[]], [[]]
end

# modification of code from Peter J. Schmid
function unstructured_tree(timeseries, p_min; threshold = 2)
    n = size(timeseries)[2]
    n_min = floor(Int, threshold * p_min * n)
    W, F, G, P1, P2 = [collect(1:n)], [], [], [1], []
    H = []
    C = Dict()
    P3 = Dict()
    P4 = Dict()
    P5 = Dict()
    CC = Dict()
    leaf_index = 1
    global_index = 1
    while (length(W) > 0)
        w = popfirst!(W)
        p1 = popfirst!(P1)
        inds, centers = split(timeseries, w, n_min)
        C[p1] =  centers
        if all([length(ind) > 0 for ind in inds])
            W = [inds..., W...]
            Ptmp = []
            [push!(Ptmp, global_index + i) for i in eachindex(inds)]
            P1 = [Ptmp..., P1...]
            [push!(P2, (p1, global_index + i, length(ind) / n)) for (i, ind) in enumerate(inds)]
            Ptmp2 = Int64[]
            [push!(Ptmp2, global_index + i) for (i, ind) in enumerate(inds)]
            [CC[global_index + i] = centers[i] for i in eachindex(inds)]
            P3[p1] = Ptmp2
            global_index += length(inds)
            push!(H, [inds...])
        else
            push!(F, w)
            push!(H, [[]])
            P3[p1] = NaN
            P4[p1] = leaf_index
            P5[leaf_index] = p1
            leaf_index += 1
        end
    end
    return F, G, H, P2, P3, P4, C, CC, P5
end

function determine_partition(timeseries, tree_type::Tree{Val{false}, S}; override = false) where S
    if haskey(tree_type.arguments, :minimum_probability)
        minimum_probability = tree_type.arguments.minimum_probability 
    elseif istype(tree_type.arguments, Number)
        minimum_probability = Tree.arguments
    else
        @warn "no minimum probability specified, using 0.01"
        minimum_probability = 0.01
    end
    Nmax = 100 * round(Int, 1/ minimum_probability)
    if (size(timeseries)[2] > Nmax) & !(override)
        @warn "timeseries too long, truncating to roughly $Nmax for determining embedding"
        skip = round(Int, size(timeseries)[2] / Nmax)
        timeseries = timeseries[:, 1:skip:end]
    end
    F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(timeseries, minimum_probability)
    embedding = UnstructuredTree(P4, C, P3)
    return embedding
end
