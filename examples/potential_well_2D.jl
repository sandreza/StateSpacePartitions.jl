using StateSpacePartitions, ProgressBars, Enzyme, GLMakie, MarkovChainHammer, LinearAlgebra, Random
Random.seed!(1234)
# Potential well test
function V(v)
    x = v[1]
    y = v[2]
    term1 = 3 * exp(−x^2 − (y − 1/3)^2) 
    term2 = -3 * exp(-x^2 - (y-5/3)^2)
    term3 = -5 * exp(-(x+1)^2 - y^2)
    term4 = -5 * exp(-(x-1)^2 - y^2)
    term5 = (x^4)/5 + ((y- 1/3)^4)/5 - y
    return term1 + term2 + term3 + term4 + term5
end

∇V(x) =  -gradient(Enzyme.Reverse, V, x)
##
ϵ = 0.7
Nₜ = 1000
Nₑ = 10
Δt = 0.05
x₀ = randn(2, Nₑ)
# spin up
for t in ProgressBar(1:Nₜ)
    𝒩 = randn(2, Nₑ)
    for ω in 1:Nₑ
        x₀[:, ω] .= x₀[:, ω] + Δt * ∇V(x₀[:, ω]) + ϵ  * √Δt * 𝒩[:, ω]
    end
end

# trajectories 
ϵ = 0.7
Δt = 0.05
Nₜ = 10^7
xₜ = randn(2, Nₜ, Nₑ)
xₜ[:, 1, :] .= x₀
for t in ProgressBar(1:Nₜ-1)
    𝒩 = randn(2, Nₑ)
    for ω in 1:Nₑ
        xₜ[:, t + 1, ω] .= xₜ[:, t, ω] + Δt * ∇V(xₜ[:, t, ω]) + ϵ  * √Δt * 𝒩[:, ω]
    end
end

trajectory = reshape(xₜ, (2, Nₜ * Nₑ))
method = Tree(false, 0.25)
state_space_partitions = StateSpacePartition(trajectory; method = method)
classified = reshape(state_space_partitions.partitions, (Nₜ, Nₑ))
pfs = [perron_frobenius(classified[:, i], step = 20) for i in ProgressBar(1:Nₑ)]
indices = size.(pfs, 1) .> (size(pfs[1], 1)- 1)
pf = sum(pfs[indices])/sum(indices)

Λ, W = eigen(pf)
μ = real.(W[:, end]) / real.(sum(W[:, end]))
Pᵣ = (pf + Diagonal(μ) * (pf') * Diagonal(1 ./ μ)) / 2
eigvals(Pᵣ)
Λᵣ, Vᵣ = eigen(Pᵣ)

# perturbations
ϵ = 0.7
Δt = 0.05
Nₑ = 4
Nₜ = 10^7
δ = [[0.05, 0.0], [-0.05, 0.0], [0.0, 0.05], [0.0, -0.05]]
xₜ = randn(2, Nₜ, Nₑ)
xₜ[:, 1, :] .= x₀[:, 1:4]
for t in ProgressBar(1:Nₜ-1)
    𝒩 = randn(2, Nₑ)
    for ω in 1:Nₑ
        xₜ[:, t + 1, ω] .= xₜ[:, t, ω] + Δt * (∇V(xₜ[:, t, ω]) + δ[ω]) + ϵ  * √Δt * 𝒩[:, ω]
    end
end

classifed_trajectories = zeros(Int, Nₜ, Nₑ)
for ω ∈ ProgressBar(1:Nₑ)
    for t ∈ ProgressBar(1:Nₜ)
        classifed_trajectories[t, ω] = state_space_partitions.embedding(xₜ[:, t, ω])
    end
end

pfs = [perron_frobenius(classifed_trajectories[:, i], step = 20) for i in ProgressBar(1:Nₑ)]
for i in 1:4
    pfp = pfs[i]
    Λ, W = eigen(pfp)
    μ = real.(W[:, end]) / real.(sum(W[:, end]))
    Pᵣ = (pfp + Diagonal(μ) * (pfp') * Diagonal(1 ./ μ)) / 2
    pfs[i] .= Pᵣ
end
mx = (pfs[1] - pfs[2]) / 0.1
my = (pfs[3] - pfs[4]) / 0.1

##
Ns = 100
xs = reshape(range(-2, 2, length = Ns), (Ns, 1))
ys = reshape(range(-2, 2, length = Ns), (1, Ns))
Vs = [V([xs[i], ys[j]]) for i in 1:Ns, j in 1:Ns]
heatmap(xs[:], ys[:], Vs, colormap = :balance, show_axis = true)

##
ϵ = 0.7
Nₜ = 1000
Nₑ = 10
Δt = 0.05
y₀ = randn(2, Nₑ)
y₀ .= x₀
# spin up
for t in ProgressBar(1:Nₜ)
    𝒩 = randn(2, Nₑ)
    for ω in 1:Nₑ
        y₀[:, ω] .= y₀[:, ω] + Δt * ∇V(y₀[:, ω]) 
    end
end

ind1 = y₀[1, :] .> 1
ind2 = y₀[1, :] .< -1
ind3 = y₀[2, :] .> 1
v₁ = mean(y₀[:, ind1], dims = 2)[:]
v₂ = mean(y₀[:, ind2], dims = 2)[:]
v₃ = mean(y₀[:, ind3], dims = 2)[:]
vs = [v₁, v₂, v₃]
coarse_classification = zeros(Int, size(trajectory, 2))
for (i, v) in ProgressBar(enumerate(eachcol(trajectory)))
    coarse_classification[i] = argmin([norm(v - w) for w in vs])
end

pf_coarse = perron_frobenius(coarse_classification, step = 20)
Λ, W = eigen(pf_coarse)
μ = real.(W[:, end]) / real.(sum(W[:, end]))
Pcᵣ = (pf_coarse + Diagonal(μ) * (pf_coarse') * Diagonal(1 ./ μ)) / 2
μᵢ = copy(μ)

classifed_coarse_trajectories = zeros(Int, 10^7, 4)
for ω ∈ ProgressBar(1:4)
    for t ∈ ProgressBar(1:10^7)
        state_index = argmin([norm(xₜ[:, t, ω] - w) for w in vs])
        classifed_coarse_trajectories[t, ω] = state_index
    end
end

pfs = [perron_frobenius(classifed_coarse_trajectories[:, i], step = 20) for i in ProgressBar(1:Nₑ)]
for i in 1:4
    pfp = pfs[i]
    Λ, W = eigen(pfp)
    μ = real.(W[:, end]) / real.(sum(W[:, end]))
    Plᵣ = (pfp + Diagonal(μ) * (pfp') * Diagonal(1 ./ μ)) / 2
    pfs[i] .= Plᵣ
end

mx = (pfs[1] - pfs[2]) / 0.1
my = (pfs[3] - pfs[4] ) / 0.1


Λˣ, Vˣ = eigen(mx)
Λʸ, Vʸ = eigen(my)

Vˣ[1, :] 
Vʸ[2, :] 