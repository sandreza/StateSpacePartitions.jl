using KernelAbstractions: @index, @kernel
using StateSpacePartitions.Architectures
using StateSpacePartitions.Architectures: architecture, total_length, convert, vector_type
using StaticArrays

struct UnstructuredTree{L, C, CH}
    leafmap::L 
    centers::C 
    children::CH
end

UnstructuredTree() = UnstructuredTree([], [], [])

function (embedding::UnstructuredTree)(state)
    current_index = 1
    while length(embedding.centers[current_index]) > 1
        local_child = argmin([norm(state .- center) for center in embedding.centers[current_index]])
        current_index = embedding.children[current_index][local_child]
    end
    return embedding.leafmap[current_index]
end

function (embedding::UnstructuredTree)(partitions, states)
    worksize  = total_length(states)
    workgroup = min(length(partitions), 256)

    arch = architecture(partitions)
    args = (partitions, states, embedding.centers, embedding.children, embedding.leafmap)

    launch_chunked_kernel!(arch, workgroup, worksize, _compute_embedding!, args)

    return nothing
end

@kernel function _compute_embedding!(partitions, states, centers, children, leafmap)
    p = @index(Global, Linear)
    
    @inbounds begin
        state = states[:, p]

        current_index = 1
        while length(centers[current_index]) > 1
            local_child = argmin([norm(state .- center) for center in centers[current_index]])
            current_index = children[current_index][local_child]
        end

        partitions[p] = leafmap[current_index]
    end
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
function unstructured_tree(trajectory, p_min; threshold = 2, architecture = CPU())
    n = size(trajectory)[2]
    n_min = floor(Int, threshold * p_min * n)
    W, F, P1, edge_information = [collect(1:n)], [], [1], []
    H = []
    centers_list = Dict{Int64, Vector{Vector{Float64}}}()
    parent_to_children = Dict{Int64, Vector{Int64}}()
    global_to_local = Dict{Int64, Int64}()
    local_to_global = Dict{Int64, Int64}()
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
            parent_to_children[p1] = Int64[]
            global_to_local[p1] = leaf_index
            local_to_global[leaf_index] = p1
            leaf_index += 1
        end
    end

    VectorType = vector_type(architecture)

    centers_list_vector       = Vector{VectorType}(undef, length(centers_list))
    parent_to_children_vector = Vector{VectorType{Int64}}(undef, length(centers_list))
    global_to_local_vector    = Vector{Int64}(undef, maximum(keys(global_to_local)))

    for n in eachindex(centers_list_vector)
        centers = centers_list[n]
        centers = Tuple.(centers)
        centers_list_vector[n] = VectorType(centers)
        parent_to_children_vector[n] = VectorType(parent_to_children[n])
    end

    # Creating the nested data structure

    for n in keys(global_to_local)
        global_to_local_vector[n] = global_to_local[n]
    end

    centers_list_vector = tuple(centers_list_vector...)
    parent_to_children_vector = tuple(parent_to_children_vector...)
    global_to_local_vector = convert(architecture, global_to_local_vector)

    return F, H, edge_information, parent_to_children_vector, global_to_local_vector, centers_list_vector, CC, local_to_global
end

"""
    determine_partition(trajectory, tree_type::Tree{Val{false}, S}; override = false) where S

# Decription

This function determines the partition of a trajectory into an unstructured tree. The tree structure is specified by the `tree_type` argument. The `trajectory` argument is a trajectory of states. The `override` keyword argument is a boolean indicating whether to override the truncation of the trajectory.

# Arguments

* `trajectory`: a trajectory of states
* `tree_type`: a `Tree` type

# Keyword Arguments

* `override`: a boolean indicating whether to override the truncation of the trajectory

# Returns

* `embedding`: a `Tree` object
"""
function determine_partition(trajectory, tree_type::Tree{Val{false}, S}; override = false, architecture = CPU()) where S
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
    F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(trajectory, minimum_probability; architecture)
    embedding = UnstructuredTree(global_to_local, centers_list, parent_to_children)
    return embedding
end
