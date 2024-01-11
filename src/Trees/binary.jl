using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

export extract_coarse_guess

struct BinaryTree{S, T}
    centers::S
    levels::T
end

function (embedding::BinaryTree)(current_state)
    global_index = 1 
    for level in 1:embedding.levels
        new_index = argmin([norm(current_state - markov_state) for markov_state in embedding.centers[global_index]])
        global_index = child_global_index(new_index, global_index)
    end
    return local_index(global_index, embedding.levels)
end

function (embedding::BinaryTree)(partitions, states)
    worksize  = total_length(states)
    workgroup = min(length(partitions), 256)

    arch = architecture(partitions)
    args = (partitions, states, embedding.centers, Val(embedding.levels))

    launch_chunked_kernel!(arch, workgroup, worksize, _compute_binary_embedding!, args)

    return nothing
end

@kernel function _compute_binary_embedding!(partitions, states, centers, ::Val{levels}) where levels
    p = @index(Global, Linear)
    
    @inbounds begin
        state = states[:, p]

        global_index = 1 

        @unroll for _ in 1:levels
            new_index = argmin([norm(state - center) for center in centers[global_index]])
            global_index = child_global_index(new_index, global_index)
        end
    
        partitions[p] = local_index(global_index, levels)
    end
end

# binary tree index juggling
local_index(global_index, levels) = global_index - 2^levels + 1 # markov index from [1, 2^levels]
child_global_index(new_index, global_parent_index) = 2 * global_parent_index + new_index - 1 
level_global_indices(level) = 2^(level-1):2^level-1
parent_global_index(child_index) = div(child_index, 2) # both global

# markov states from centers list 
function get_markov_states(centers_list::Vector{Vector{Vector{Float64}}}, level)
    markov_states = Vector{Float64}[]
    indices = level_global_indices(level)
    for index in indices
        push!(markov_states, centers_list[index][1])
        push!(markov_states, centers_list[index][2])
    end
    return markov_states
end
get_markov_states(embedding::BinaryTree, level) = get_markov_states(embedding.markov_states, level)

function binary_split(trajectory)
    numstates = 2
    r0 = kmeans(trajectory, numstates; max_iters=10000)
    child_0 = (r0.assignments .== 1)
    child_1 = (!).(child_0)
    children = [view(trajectory, :, child_0), view(trajectory, :, child_1)]
    return r0.centers, children
end

function binary_tree(trajectory, levels)
    parent_views = []
    centers_list = Vector{Vector{Float64}}[]
    push!(parent_views, trajectory)
    ## Level 1
    centers, children = binary_split(trajectory)
    push!(centers_list, [centers[:, 1], centers[:, 2]])
    push!(parent_views, children[1])
    push!(parent_views, children[2])
    ## Levels 2 through levels
    for level in 2:levels
        for parent_global_index in level_global_indices(level)
            centers, children = binary_split(parent_views[parent_global_index])
            push!(centers_list, [centers[:, 1], centers[:, 2]])
            push!(parent_views, children[1])
            push!(parent_views, children[2])
        end
    end
    return centers_list, levels
end

function extract_coarse_guess(coarse_pfo, levels, index)
    ll, vv = eigen(coarse_pfo)
    guess = zeros(ComplexF64 ,size(coarse_pfo)[1] * 2^levels)
    for i in eachindex(guess)
        guess[i] = vv[(i-1)÷(2^levels) + 1, index]
    end
    return ll[index], guess / norm(guess)
end

"""
    determine_partition(trajectory, tree_type::Tree{Val{true}, S}; override = false) where S

# Decription

This function determines the partitioning of the state space into a binary tree. The tree is determined by recursively splitting the state space into two parts. The splitting is done by k-means clustering. The number of levels of the tree is determined by the `levels` keyword argument. If the trajectory is too long, it is truncated to roughly 1000 points per level. The `override` keyword argument can be used to override this behavior.

# Arguments

* `trajectory`: a trajectory of states
* `tree_type`: a `Tree` type

# Keyword Arguments

* `override`: a boolean indicating whether to override the truncation of the trajectory

# Returns

* `embedding`: a `Tree` object
"""
function determine_partition(trajectory, tree_type::Tree{Val{true}, S}; override = false, kwargs...) where S
    if typeof(tree_type.arguments) <: NamedTuple
        if haskey(tree_type.arguments, :levels)
            levels = tree_type.arguments.levels
        else
            @warn "no levels specified, using 7"
            levels = 7
        end
    elseif typeof(tree_type.arguments) <: Number
        levels = tree_type.arguments
    else
        @warn "no levels specified, using 7"
        levels = 7
    end
    Nmax = 1000 * round(Int, 2^levels)
    if (size(trajectory)[2] > Nmax) & !(override)
        @warn "trajectory too long, truncating to roughly $Nmax for determining embedding"
        skip = round(Int, size(trajectory)[2] / Nmax)
        trajectory = trajectory[:, 1:skip:end]
    end
    centers_list, levels = binary_tree(trajectory, levels)
    embedding = BinaryTree(centers_list, levels)
    return embedding
end
