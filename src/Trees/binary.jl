export extract_coarse_guess
struct BinaryTree{S, T}
    markov_states::S
    levels::T
end

function (embedding::BinaryTree)(current_state)
    global_index = 1 
    for level in 1:embedding.levels
        new_index = argmin([norm(current_state - markov_state) for markov_state in embedding.markov_states[global_index]])
        global_index = child_global_index(new_index, global_index)
    end
    return local_index(global_index, embedding.levels)
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

function binary_split(timeseries)
    numstates = 2
    r0 = kmeans(timeseries, numstates; max_iters=10000)
    child_0 = (r0.assignments .== 1)
    child_1 = (!).(child_0)
    children = [view(timeseries, :, child_0), view(timeseries, :, child_1)]
    return r0.centers, children
end

function binary_tree(timeseries, levels)
    parent_views = []
    centers_list = Vector{Vector{Float64}}[]
    push!(parent_views, timeseries)
    ## Level 1
    centers, children = binary_split(timeseries)
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

function determine_partition(timeseries, tree_type::Tree{Val{true}, S}; override = false) where S
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
    if (size(timeseries)[2] > Nmax) & !(override)
        @warn "timeseries too long, truncating to roughly $Nmax for determining embedding"
        skip = round(Int, size(timeseries)[2] / Nmax)
        timeseries = timeseries[:, 1:skip:end]
    end
    centers_list, levels = binary_tree(timeseries, levels)
    embedding = BinaryTree(centers_list, levels)
    return embedding
end
