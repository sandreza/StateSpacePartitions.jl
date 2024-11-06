using Distributions

struct StatisticalModel{E, M, T}
    embedding::E
    invariant_measure::M
    transition_operator::T 
end

struct DeltaFunction{W, C}
    weights::W
    means::C
end

struct GeneralGaussianMixture{W, M, C}
    weights::W
    means::M
    covariances::C
end


function determine_statistical_model(trajectory, tree_type::Tree{Val{false},S}; override=false) where {S}
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
    Nmax = 100 * round(Int, 1 / minimum_probability)
    c_trajectory = copy(trajectory)
    if (size(trajectory)[2] > Nmax) & !(override)
        @warn "trajectory too long, truncating to roughly $Nmax for determining embedding"
        skip = round(Int, size(trajectory)[2] / Nmax)
        c_trajectory = copy(trajectory[:, 1:skip:end])
    end
    if (10 / (size(trajectory)[2]) > minimum_probability) & !(override)
        @warn "minimum probabity too small, using 10x the reciprocal of the number of points"
        minimum_probability = 10 / size(trajectory)[2]
    end
    F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(c_trajectory, minimum_probability)

    embedding = UnstructuredTree(global_to_local, centers_list, parent_to_children)

    means = zeros(size(trajectory, 1), length(global_to_local))
    for key in keys(global_to_local)
        local_index = global_to_local[key]
        means[:, local_index] = CC[key]
    end
    # means = [means[:, i] for i in 1:size(means, 2)]
    @info "computing partition trajectory"
    probability_weights = zeros(length(means))
    partitions = zeros(Int64, size(trajectory)[2])
    for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
        cell_index = embedding(state)
        partitions[i] = cell_index
        probability_weights[cell_index] += 1 / size(trajectory)[2]
    end
    probability_weights = probability_weights / sum(probability_weights)

    return embedding, probability_weights, means, partitions
end


# include the lorenz file here
method = Tree(false, 0.01)

emb, pr, ms, partitions = determine_statistical_model(trajectory, method)

empirical_centers = zeros(size(trajectory)[1], maximum(partitions))
empirical_covariance = zeros(size(trajectory)[1], size(trajectory)[1], maximum(partitions))
empirical_count = zeros(Int64, maximum(partitions))
for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
    cell_index = embedding(state)
    partitions[i] = cell_index
    empirical_count[cell_index] += 1
    empirical_centers[:, cell_index] .+= state
    empirical_covariance[:, :, cell_index] .+= state * state'
end
# adjust 
adj_empirical_centers = empirical_centers ./ reshape(empirical_count, 1, length(empirical_count))
adj_empirical_covariance = empirical_covariance ./ reshape(empirical_count .- 1, 1, 1, length(empirical_count))
for i in 1:length(probability_weights)
    adj_empirical_covariance[:, :, i] .-= (adj_empirical_centers[:, i] * adj_empirical_centers[:, i]') * empirical_count[i] / (empirical_count[i] - 1)
end
probability_weights = empirical_count / sum(empirical_count)

δmodel = DeltaFunction(probability_weights, adj_empirical_centers)
Σmodel = GeneralGaussianMixture(probability_weights, adj_empirical_centers, adj_empirical_covariance)

import Base.rand
import Statistics.mean
import Statistics.cov

function rand(δmodel::DeltaFunction)
    cell_index = rand(Categorical(δmodel.weights))
    return δmodel.means[:, cell_index]
end

function rand(Σmodel::GeneralGaussianMixture)
    cell_index = rand(Categorical(Σmodel.weights))
    return rand(MvNormal(Σmodel.means[:, cell_index], Σmodel.covariances[:, :, cell_index]))
end

function mean(δmodel::DeltaFunction)
    ensemble_mean = zeros(size(δmodel.means)[1])
    for i in 1:size(δmodel.means)[2]
        ensemble_mean .+= δmodel.weights[i] * δmodel.means[:, i]
    end
    return ensemble_mean
end

function mean(Σmodel::GeneralGaussianMixture)
    ensemble_mean = zeros(size(Σmodel.means)[1])
    for i in 1:size(Σmodel.means)[2]
        ensemble_mean .+= Σmodel.weights[i] * Σmodel.means[:, i]
    end
    return ensemble_mean
end

function std(δmodel::DeltaFunction)
    ensemble_std = zeros(size(δmodel.means)[1])
    ensemble_mean = mean(δmodel)
    for i in 1:size(δmodel.means)[2]
        ensemble_std .+= δmodel.weights[i] * (δmodel.means[:, i] .- ensemble_mean).^2
    end
    return sqrt.(ensemble_std)
end

function cov(δmodel::DeltaFunction)
    ensemble_cov = zeros(size(δmodel.means)[1], size(δmodel.means)[1])
    ensemble_mean = mean(δmodel)
    for i in 1:size(δmodel.means)[2]
        ensemble_cov .+= δmodel.weights[i] * (δmodel.means[:, i] .- ensemble_mean) * (δmodel.means[:, i] .- ensemble_mean)'
    end
    return ensemble_cov
end

function cov(Σmodel::GeneralGaussianMixture)
    ensemble_cov = zeros(size(Σmodel.means)[1], size(Σmodel.means)[1])
    for i in 1:size(Σmodel.means)[2]
        ensemble_cov .+= Σmodel.weights[i] * (Σmodel.covariances[:, :, i] + Σmodel.means[:,i] * Σmodel.means[:,i]')
    end
    ensemble_mean = mean(Σmodel)
    ensemble_cov .-= ensemble_mean * ensemble_mean'
    return ensemble_cov
end

indval = 2
ind1 = trajectory[:, partitions .== indval]
check1 = mean(ind1, dims=2)
adj_empirical_centers[:, indval] - check1
cov1 = cov(ind1')
adj_empirical_covariance[:, :, indval]

Nsamples = 100
samples = zeros(size(trajectory)[1], Nsamples)
for i in 1:Nsamples
    samples[:, i] = rand(Σmodel)
end