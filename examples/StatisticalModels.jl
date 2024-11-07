using Distributions, StateSpacePartitions
import Base.rand
import Statistics.mean
import Statistics.cov


struct StatisticalModel{E, M, T}
    embedding::E
    invariant_measure::M
    transition_operator::T 
end

struct DeltaFunction{W, C}
    weights::W
    means::C
end

function DeltaFunction(trajectory::Matrix, cells::Int)
    state_space_partition = StateSpacePartition(trajectory; cells)
    empirical_centers = zeros(size(trajectory)[1], maximum(state_space_partition.partitions))
    empirical_count = zeros(Int64, maximum(state_space_partition.partitions))
    for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
        cell_index = state_space_partition.partitions[i]
        partitions[i] = cell_index # overwritten
        empirical_count[cell_index] += 1
        empirical_centers[:, cell_index] .+= state
    end
    adj_empirical_centers = empirical_centers ./ reshape(empirical_count, 1, length(empirical_count))
    probability_weights = empirical_count / sum(empirical_count)
    return DeltaFunction(probability_weights, adj_empirical_centers)
end

function DeltaFunction(state_space_partition::StateSpacePartition)
    empirical_centers = zeros(size(trajectory)[1], maximum(state_space_partition.partitions))
    empirical_count = zeros(Int64, maximum(state_space_partition.partitions))
    for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
        cell_index = state_space_partition.partitions[i]
        partitions[i] = cell_index # overwritten
        empirical_count[cell_index] += 1
        empirical_centers[:, cell_index] .+= state
    end
    adj_empirical_centers = empirical_centers ./ reshape(empirical_count, 1, length(empirical_count))
    probability_weights = empirical_count / sum(empirical_count)
    return DeltaFunction(probability_weights, adj_empirical_centers)
end

function rand(δmodel::DeltaFunction)
    cell_index = rand(Categorical(δmodel.weights))
    return δmodel.means[:, cell_index]
end

function rand(δmodel::DeltaFunction, N::Int)
    samples = zeros(size(δmodel.means)[1], N)
    for i in 1:N 
        samples[:, i] .= rand(δmodel)
    end
    return samples
end

function mean(δmodel::DeltaFunction)
    ensemble_mean = zeros(size(δmodel.means)[1])
    for i in 1:size(δmodel.means)[2]
        ensemble_mean .+= δmodel.weights[i] * δmodel.means[:, i]
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


struct GaussianMixture{W, M, C}
    weights::W
    means::M
    covariances::C
end

function GaussianMixture(trajectory::Matrix, cells::Int)
    state_space_partition = StateSpacePartition(trajectory; cells)
    empirical_centers = zeros(size(trajectory)[1], maximum(state_space_partition.partitions))
    empirical_covariance = zeros(size(trajectory)[1], size(trajectory)[1], maximum(state_space_partition.partitions))
    empirical_count = zeros(Int64, maximum(state_space_partition.partitions))
    for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
        cell_index = state_space_partition.partitions[i]
        partitions[i] = cell_index # overwritten
        empirical_count[cell_index] += 1
        empirical_centers[:, cell_index] .+= state
        empirical_covariance[:, :, cell_index] .+= state * state'
    end
    adj_empirical_centers = empirical_centers ./ reshape(empirical_count, 1, length(empirical_count))
    adj_empirical_covariance = empirical_covariance ./ reshape(empirical_count .- 1, 1, 1, length(empirical_count))
    for i in 1:length(empirical_count)
        adj_empirical_covariance[:, :, i] .-= (adj_empirical_centers[:, i] * adj_empirical_centers[:, i]') * empirical_count[i] / (empirical_count[i] - 1)
    end
    probability_weights = empirical_count / sum(empirical_count)
    return GaussianMixture(probability_weights, adj_empirical_centers, adj_empirical_covariance)
end

function GaussianMixture(state_space_partition::StateSpacePartition, trajectory::Matrix)
    empirical_centers = zeros(size(trajectory)[1], maximum(state_space_partition.partitions))
    empirical_covariance = zeros(size(trajectory)[1], size(trajectory)[1], maximum(state_space_partition.partitions))
    empirical_count = zeros(Int64, maximum(state_space_partition.partitions))
    for (i, state) in ProgressBar(enumerate(eachcol(trajectory)))
        cell_index = state_space_partition.partitions[i]
        partitions[i] = cell_index # overwritten
        empirical_count[cell_index] += 1
        empirical_centers[:, cell_index] .+= state
        empirical_covariance[:, :, cell_index] .+= state * state'
    end
    adj_empirical_centers = empirical_centers ./ reshape(empirical_count, 1, length(empirical_count))
    adj_empirical_covariance = empirical_covariance ./ reshape(empirical_count .- 1, 1, 1, length(empirical_count))
    for i in 1:length(empirical_count)
        adj_empirical_covariance[:, :, i] .-= (adj_empirical_centers[:, i] * adj_empirical_centers[:, i]') * empirical_count[i] / (empirical_count[i] - 1)
    end
    probability_weights = empirical_count / sum(empirical_count)
    return GaussianMixture(probability_weights, adj_empirical_centers, adj_empirical_covariance)
end

function (score::GaussianMixture)(x)
    n = size(GaussianMixture.means)[1]
    m = size(GaussianMixture.means)[2]
    score_value = zeros(n)
    denominator = [0.0]
    for i in 1:m
        Δ = GaussianMixture.means[:, i] - x
        Σ⁻¹Δ = score.inverse_covariances[:, :, i] * Δ
        normalization = sqrt(det(2π * score.probability_model.covariances[:, :, i]))
        U = exp(-0.5 * Δ' * Σ⁻¹Δ) / normalization
        weightedU = score.probability_model.weights[i] * U 
        probability_value .+= weightedU 
    end
    return probability_value
end

function isotropic_inflate!(Σmodel::GaussianMixture, factor::Real)
    for i in 1:size(Σmodel.covariances)[3]
        Σmodel.covariances[:, :, i] .= Σmodel.covariances[:, :, i] + factor * I
    end
end

function general_inflate!(Σmodel::GaussianMixture, inflation_matrix::Matrix{Float64})
    for i in 1:size(Σmodel.covariances)[3]
        Σmodel.covariances[:, :, i] .+= inflation_matrix
    end
end

function scaled_inflate!(Σmodel::GaussianMixture, factor::Real)
    for i in 1:size(Σmodel.covariances)[3]
        Σmodel.covariances[:, :, i] .*= factor
    end
end

function average_covariance(Σmodel::GaussianMixture)
    average_covariance = similar(Σmodel.covariances[:, :, 1]) * 0
    for i in eachindex(Σmodel.weights)
        average_covariance .+= Σmodel.weights[i] * Σmodel.covariances[:, :, i]
    end
    return average_covariance
end

function rand(Σmodel::GaussianMixture)
    cell_index = rand(Categorical(Σmodel.weights))
    return rand(MvNormal(Σmodel.means[:, cell_index], Σmodel.covariances[:, :, cell_index]))
end

function rand(Σmodel::GaussianMixture, N::Int)
    samples = zeros(size(Σmodel.means)[1], N)
    for i in 1:N 
        samples[:, i] .= rand(Σmodel)
    end
    return samples
end

function rand_fixed_cell(Σmodel::GaussianMixture, cell_index::Int)
    return rand(MvNormal(Σmodel.means[:, cell_index], Σmodel.covariances[:, :, cell_index]))
end

"""
    samples_fixed_cell(Σmodel::GaussianMixture, cell_index::Int, N::Int)
    
    Generate N samples from a fixed cell with cell index = cell_index in a Gaussian mixture model.
"""
function samples_fixed_cell(Σmodel::GaussianMixture, cell_index::Int, N::Int)
    samples = zeros(size(Σmodel.means)[1], N)
    for i in 1:N 
        samples[:, i] .= rand_fixed_cell(Σmodel, cell_index)
    end
    return samples
end

function mean(Σmodel::GaussianMixture)
    ensemble_mean = zeros(size(Σmodel.means)[1])
    for i in 1:size(Σmodel.means)[2]
        ensemble_mean .+= Σmodel.weights[i] * Σmodel.means[:, i]
    end
    return ensemble_mean
end

function cov(Σmodel::GaussianMixture)
    ensemble_cov = zeros(size(Σmodel.means)[1], size(Σmodel.means)[1])
    for i in 1:size(Σmodel.means)[2]
        ensemble_cov .+= Σmodel.weights[i] * (Σmodel.covariances[:, :, i] + Σmodel.means[:,i] * Σmodel.means[:,i]')
    end
    ensemble_mean = mean(Σmodel)
    ensemble_cov .-= ensemble_mean * ensemble_mean'
    return ensemble_cov
end

struct ScoreModel{P, S}
    probability_model::P
    inverse_covariances::S
end

function ScoreModel(gm::GaussianMixture)
    n = size(gm.means)[1]
    m = size(gm.means)[2]
    Σinv = zeros(n, n, m)
    for i in 1:m
        Σinv[:, :, i] = pinv(gm.covariances[:, :, i])
    end
    return ScoreModel(gm, Σinv)
end

function (score::ScoreModel)(x)
    n = size(score.probability_model.means)[1]
    m = size(score.probability_model.means)[2]
    score_value = zeros(n)
    denominator = [0.0]
    for i in 1:m
        Δ = score.probability_model.means[:, i] - x
        Σ⁻¹Δ = score.inverse_covariances[:, :, i] * Δ
        normalization = sqrt(det(2π * score.probability_model.covariances[:, :, i]))
        U = exp(-0.5 * Δ' * Σ⁻¹Δ) / normalization
        weightedU = score.probability_model.weights[i] * U 
        score_value .+= weightedU * Σ⁻¹Δ
        denominator .+= weightedU
    end
    return score_value / denominator[1]
end

function (score::ScoreModel)(x, sigma)
    n = size(score.probability_model.means)[1]
    m = size(score.probability_model.means)[2]
    score_value = zeros(n)
    denominator = [0.0]
    for i in 1:m
        Δ = score.probability_model.means[:, i] - x
        U = exp(-(0.5 /sigma^2) * Δ' * Δ)
        weightedU = score.probability_model.weights[i] * U
        score_value .+= weightedU * Δ
        denominator .+= weightedU
    end
    return score_value / ( denominator[1] * sigma^2)
end

