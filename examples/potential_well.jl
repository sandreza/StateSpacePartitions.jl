# Potential well test
using StateSpacePartitions, ProgressBars, Enzyme, GLMakie, MarkovChainHammer, LinearAlgebra, Random
function V(x)
    return (x[1]^2 - 1)^2 # + x[1] * 0.1
end
include("StatisticalModels.jl")
∇V(x) =  -gradient(Enzyme.Reverse, V, x)
##
ϵ = sqrt(2)
Nₜ = 1000
Nₑ = 100000
Δt = 0.01
x₀ = randn(Nₑ)
for t in ProgressBar(1:Nₜ)
    𝒩 = randn(Nₑ)
    for ω in 1:Nₑ
        x₀[ω] = x₀[ω] + Δt * ∇V([x₀[ω]])[1] + ϵ  * √Δt * 𝒩[ω]
    end
end
hist(x₀)

xs = range(-3, 3, length = 1000)
unnormalized_density = exp.(-V.(-xs))
Z = sum(unnormalized_density) * (xs[end] - xs[end-1])
normalized_density = unnormalized_density / Z

fig = Figure() 
ax = Axis(fig[1,1]) 
GLMakie.density!(ax, x₀[:], color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
lines!(ax, xs, normalized_density)
display(fig)
##
σ = 0.01

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "y", ylabel = "- Z / σ")
Z = randn(Nₑ)
scatter!(ax, x₀ + σ * Z,  - Z / σ, color = (:red, 0.01))
display(fig)


GaussianMixture()
##
cells = round(Int, 30 * 1.5)
ssp = StateSpacePartition(reshape(x₀, (1, Nₑ)); cells)
##
Σmodel = GaussianMixture(ssp, reshape(x₀, (1, Nₑ)))
score = ScoreModel(Σmodel)
exact_score = [∇V(x) for x in x₀]
gmm_score = [score([x])[1] for x in x₀]
gmm_score_r = [mean([score([x + 0.1 * randn()])[1] for i in 1:10]) for x in ProgressBar(x₀)]
lines(sort(x₀), gmm_score[sortperm(x₀)], color = (:blue, 0.5), linewidth = 3, label = "GM score")
scatter!(sort(Σmodel.means[:]), gmm_score[sortperm(Σmodel.means[:])], color = :orange, markersize = 10, label = "GM means")
lines!(sort(x₀), gmm_score_r[sortperm(x₀)], color = (:blue, 0.5), linewidth = 3, label = "GM score R")
scatter!(sort(Σmodel.means[:]), gmm_score_r[sortperm(Σmodel.means[:])], color = :orange, markersize = 10, label = "GM means R")
fig = Figure(resolution = (1200, 1200))

for (k, σ) in enumerate([0.01, 0.05, 0.1, 0.5])
    ii = (k-1)÷2 + 1
    jj = (k-1) % 2 + 1
    x⃗ = zeros(2, Nₑ)
    x⃗[1, :] = x₀ + σ * Z
    x⃗[2, :] = - Z / σ
    ssp_sigma = StateSpacePartition(x⃗[1:1, :]; cells)
    Nc = maximum(ssp_sigma.partitions)
    yC = zeros(2, Nc)
    for i in 1:Nc
        yC[:, i] .= mean(x⃗[:, ssp_sigma.partitions .== i], dims = 2)[:, 1]
    end
    gm_score = [score([x₀[i]], σ)[1] for i in 1:Nₑ]
    ax = Axis(fig[ii, jj]; xlabel = "x", ylabel = "Score", title = "Score comparison,  σ = $σ")
    lines!(ax, sort(x₀), gm_score[sortperm(x₀)], color = (:blue, 0.5), linewidth = 3, label = "GM score")
    lines!(ax, sort(x₀), exact_score[sortperm(x₀)], color = (:red, 0.5), linewidth = 3, label = "Exact score")
    scatter!(ax, yC[1, :], yC[2, :], color = :green, label = "Ludo Score")
    if k == 4
        axislegend(ax, position = :rt, orientation = :vertical)
    end
    xlims!(-2, 2)
    ylims!(-20, 20)
    display(fig)
end


##
# gm = GaussianMixture(reshape(x₀, (1, Nₑ)), 10000)

y₀ = x₀[1:1:end]
covy0 = reshape(ones(length(y₀)), (1, 1, length(y₀)))
gm = GaussianMixture([1/length(y₀) for i in eachindex(y₀)], reshape(y₀, (1, length(y₀))), covy0)
score = ScoreModel(gm)
xs = range(-2, 2, length = 200)
fig = Figure()
for (i, σ) in enumerate([0.1, 0.5, 1.0, 2])
    ii = (i-1)÷2 + 1
    jj = (i-1) % 2 + 1
    ax = Axis(fig[ii, jj], xlabel = "x", ylabel = "Score", title = "Score comparison,  σ = $σ")
    lines!(ax, xs, [score([x], σ)[1] for x in xs], color = (:blue, 0.5), linewidth = 3, label = "GM score")
    lines!(ax, xs, [∇V([x])[1] for x in xs], color = (:red, 0.5), linewidth = 3, label = "Exact score")
    xlims!(ax, -2, 2)
    ylims!(ax, -20, 20)
end
display(fig)

##
gm = GaussianMixture(reshape(x₀, (1, Nₑ)), 10000)
score = ScoreModel(gm)
xs = range(-2, 2, length = 200)
fig = Figure()
for (i, σ) in enumerate([0.1, 0.5, 1.0, 2])
    ii = (i-1)÷2 + 1
    jj = (i-1) % 2 + 1
    ax = Axis(fig[ii, jj], xlabel = "x", ylabel = "Score", title = "Score comparison,  σ = $σ")
    lines!(ax, xs, [score([x], σ)[1] for x in xs], color = (:blue, 0.5), linewidth = 3, label = "GM score")
    lines!(ax, xs, [∇V([x])[1] for x in xs], color = (:red, 0.5), linewidth = 3, label = "Exact score")
    xlims!(ax, -2, 2)
    ylims!(ax, -20, 20)
end
display(fig)

##
ts = range(0, 1, length = 100)
sigma_val(t) = 0.1 * exp(3*t)
sigmas = sigma_val.(reverse(ts))
diffusion_dynamics = [score([x], sigma)[1] for x in xs, sigma in sigmas]

heatmap(xs, ts, diffusion_dynamics)
##
nn(x, σ)= score([x], σ)[1] * σ^2 + x
##
σ = sigma_val(1) 
tmp = nn(randn()* σ, σ)
σ = sigma_val(0.8) 
tmp = nn(tmp, σ)
σ = sigma_val(0.6) 
tmp = nn(tmp, σ)
σ = sigma_val(0.4) 
tmp = nn(tmp, σ)
σ = sigma_val(0.2) 
tmp = nn(tmp, σ)
for i in 1:1000
    tmp = nn(tmp, 0.01)
end
##


# scaled_inflate!(Σmodel, 100)
sigma = 0.1 # * var(x₀') # var(x₀') / sqrt(cells)
general_inflate!(Σmodel, reshape([sigma], (1, 1)))
#=
covavg = average_covariance(Σmodel)
covtruth = cov(x₀)
covmodel = cov(Σmodel)
determinant_inflate!(Σmodel)
covmodel = cov(Σmodel)
scaled_inflate!(Σmodel, 2)
covmodel = cov(Σmodel)
=#
# δmin = mean(abs.(sort(Σmodel.means[:])[2:end] - sort(Σmodel.means[:])[1:end-1]))
# isotropic_inflate!(Σmodel, maximum([δmin, covtruth * 0.05]))
# scale = covtruth ./ covavg * 0.01
# scaled_inflate!(Σmodel, scale[1])
# general_inflate!(Σmodel, reshape([covavg], (1, 1)))
# average_covariance(Σmodel)
# general_inflate!(Σmodel, reshape([cov(x₀)], (1, 1)) * 0.01)

fig = Figure() 
ax = Axis(fig[1,1]) 
GLMakie.density!(ax, x₀[:], color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
tmp = [Σmodel([x]) for x in xs]
pdf_on_means = [Σmodel([x]) for x in Σmodel.means][:]
GLMakie.lines!(ax, xs, tmp, color = :blue, strokecolor = :blue,  strokewidth = 3)
GLMakie.scatter!(ax, Σmodel.means[:], pdf_on_means, color = :orange, markersize = 10)
display(fig)
##
zlist = randn(1000)
scorevals = mean([score([0 + z * 0.1]) for z in zlist])
##
score = ScoreModel(Σmodel)

model_score_values = [score([x])[1] for x in xs]
# mollified_model_score_values = [score([x], 0.01)[1] for x in xs]
model_score_values_on_data = [score([x])[1] for x in x₀[1:100:end]]
model_score_values_on_mean = [score([x])[1] for x in  Σmodel.means[:]]
exact_score_values = [∇V([x])[1] for x in xs]

fig = Figure() 
ax = Axis(fig[1,1])
xlims!(ax, -1.2, 1.2)
ylims!(ax, -2, 2)
lines!(ax, xs, exact_score_values, color = :red)
# lines!(ax, xs, mollified_model_score_values, color = :green)
lines!(ax, xs, model_score_values, color = :blue)
scatter!(ax, Σmodel.means[:], model_score_values_on_mean, color = :orange)

##

fig = Figure()
ax = Axis(fig[1, 1])
cell_number = 1
fixed_cell_samples_1 = samples_fixed_cell(Σmodel, cell_number, 1276)[1, :]
fixed_cell_samples_trajectory_1 = x₀[ssp.partitions .== cell_number]
GLMakie.density!(ax, fixed_cell_samples_1, color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
hist!(ax, fixed_cell_samples_trajectory_1, color = (:blue, 0.5), bins = 100, normalization = :pdf)
cell_number = 2
fixed_cell_samples_2 = samples_fixed_cell(Σmodel, cell_number, 1276)[1, :]
fixed_cell_samples_trajectory_2 = x₀[ssp.partitions .== cell_number]
GLMakie.density!(ax, fixed_cell_samples_2, color = (:orange, 0.1), strokecolor = :orange,  strokewidth = 3)
hist!(ax, fixed_cell_samples_trajectory_2, color = (:green, 0.5), bins = 100, normalization = :pdf)
ax = Axis(fig[1, 2])
# combine 
hist!(ax, vcat(fixed_cell_samples_trajectory_1, fixed_cell_samples_trajectory_2), color = (:blue, 0.5), bins = 100, normalization = :pdf)
GLMakie.density!(ax, vcat(fixed_cell_samples_1, fixed_cell_samples_2), color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
display(fig)

##
fig = Figure()
colors = [:red, :blue, :green, :orange, :purple, :yellow]
ax = Axis(fig[1, 1])
for cell_number in eachindex(Σmodel.means)
    fixed_cell_samples_1 = samples_fixed_cell(Σmodel, cell_number, 1000)[1, :]
    fixed_cell_samples_trajectory_1 = x₀[ssp.partitions .== cell_number]
    GLMakie.density!(ax, fixed_cell_samples_1, color = (:red, 0.1), strokecolor = (:red, 0.5), strokewidth = 3)
    # hist!(ax, fixed_cell_samples_trajectory_1, color = (:blue, 0.5), bins = 100, normalization = :pdf)
end
ax = Axis(fig[1, 2])
GLMakie.lines!(ax, xs, tmp, color = :blue, strokecolor = :blue,  strokewidth = 3)
# GLMakie.density!(ax, rand(Σmodel, 10^4)[:], color = (:blue, 0.1), strokecolor = (:blue, 0.5), strokewidth = 4)
display(fig)

##
# scaled_inflate!(Σmodel, 10.0)
# mixture_model_samples = rand(Σmodel, 10^6)[:]
isotropic_inflate!(Σmodel, 0.01)
tmp = [Σmodel([x]) for x in xs]
##
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, xs, tmp)
display(fig)


##
dim = 10000
samples = 1000
Z = randn(dim, samples)
Z2 = randn(dim, samples)
Z3 = (rand(dim, samples) .- 0.5) * sqrt(12)

nroms = [norm(Z[:, i]) for i in 1:samples]
nroms2 = [norm(Z2[:, i]) for i in 1:samples]
nroms3 = [norm(Z3[:, i]) for i in 1:samples]
diff = [norm(Z[:, i] - Z2[:, i]) for i in 1:samples]
diff2 = [norm(Z[:, i] - Z3[:, i]) for i in 1:samples]

hist(nroms)
hist!(diff)
hist!(diff2)