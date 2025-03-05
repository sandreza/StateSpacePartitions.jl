# Potential well test
using StateSpacePartitions, ProgressBars, Enzyme, GLMakie, MarkovChainHammer, LinearAlgebra, Random
Random.seed!(12345)
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

y₀ = x₀[1:2000:end]
σ = 0.0001
covy0 = reshape(ones(length(y₀)) * σ, (1, 1, length(y₀)))
gm = GaussianMixture([1/length(y₀) for i in eachindex(y₀)], reshape(y₀, (1, length(y₀))), covy0)
xs = range(-2, 2, length = 10000)
pdf = [gm([x]) for x in xs]
lines(xs, pdf)

##
y₀ = x₀[1:200:end]
σ = 0.0001
covy0 = reshape(ones(length(y₀)) * σ, (1, 1, length(y₀)))
gm = GaussianMixture([1/length(y₀) for i in eachindex(y₀)], reshape(y₀, (1, length(y₀))), covy0)
xs = range(-2, 2, length = 10000)
pdf = [gm([x]) for x in xs]
lines(xs, pdf)
##
xs_2 = range(-3, 3, length = 10000)
unnormalized_density = exp.(-V.(-xs_2))
Z = sum(unnormalized_density) * (xs_2[end] - xs_2[end-1])
normalized_density = unnormalized_density / Z
##
y₀ = x₀[1:10:end]
fig = Figure() 
sigmas = [2.0^(i-8) for i in 1:9]
for (i, sigma) in enumerate(sigmas) 
    ii = (i-1)÷3 + 1
    jj = (i-1) % 3 + 1
    ax = Axis(fig[ii, jj], xlabel = "x", ylabel = "Density", title = "σ = $sigma")
    σ = sigma
    covy0 = reshape(ones(length(y₀)) * σ, (1, 1, length(y₀)))
    gm = GaussianMixture([1/length(y₀) for i in eachindex(y₀)], reshape(y₀, (1, length(y₀))), covy0)
    xs = range(-4, 4, length = 1000)
    pdf = [gm([x]) for x in xs]
    lines!(ax, xs, pdf, color = (:blue, 0.5), linewidth = 3, label = "GM density")
    lines!(ax, xs_2, normalized_density, color = (:red, 0.5), linewidth = 3, label = "Exact density")
    if i == 1
        axislegend(ax, position = :rt)
    end
    xlims!(ax, -4, 4)
end
display(fig)

##


fig = Figure() 
ax = Axis(fig[1,1], xlabel = "x", ylabel = "Density", title = "Potential well")

xs = range(-3, 3, length = 10000)
unnormalized_density = exp.(-V.(-xs))
Z = sum(unnormalized_density) * (xs[end] - xs[end-1])
normalized_density = unnormalized_density / Z

fig = Figure() 
ax = Axis(fig[1,1]) 
GLMakie.density!(ax, x₀[:], color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
lines!(ax, xs, normalized_density)
display(fig)

##
score = ScoreModel(gm)
xs = range(-2, 2, length = 200)
fig = Figure()
for (i, sigma) in enumerate(sigmas) 
    ii = (i-1)÷3 + 1
    jj = (i-1) % 3 + 1
    σ = sigma
    ax = Axis(fig[ii, jj], xlabel = "x", ylabel = "Score", title = "Score comparison,  σ = $σ")
    lines!(ax, xs, [score([x], σ)[1] for x in xs], color = (:blue, 0.5), linewidth = 3, label = "GM score")
    lines!(ax, xs, [∇V([x])[1] for x in xs], color = (:red, 0.5), linewidth = 3, label = "Exact score")
    xlims!(ax, -2, 2)
    ylims!(ax, -10, 10)
end
display(fig)

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
x⃗s = []
gm_scores = []
exact_scores = []
yCs = []
σs = [0.01, 0.05, 0.1, 0.5]
for (k, σ) in enumerate(σs)
    ii = (k-1)÷2 + 1
    jj = (k-1) % 2 + 1
    x⃗ = zeros(2, Nₑ)
    Z = randn(Nₑ)
    x⃗[1, :] = x₀ + σ * Z
    x⃗[2, :] = - Z / σ
    push!(x⃗s, x⃗)
    ssp_sigma = StateSpacePartition(x⃗[1:1, :]; cells)
    Nc = maximum(ssp_sigma.partitions)
    yC = zeros(2, Nc)
    for i in 1:Nc
        yC[:, i] .= mean(x⃗[:, ssp_sigma.partitions .== i], dims = 2)[:, 1]
    end
    gm_score = [score([x₀[i]], σ)[1] for i in 1:Nₑ]
    push!(gm_scores, gm_score)
    push!(yCs, yC)
    ax = Axis(fig[ii, jj]; xlabel = "x", ylabel = "Score", title = "Score comparison,  σ = $σ")
    lines!(ax, sort(x₀), gm_score[sortperm(x₀)], color = (:blue, 0.5), linewidth = 3, label = "GM score")
    lines!(ax, sort(x₀), exact_score[sortperm(x₀)], color = (:red, 0.5), linewidth = 3, label = "Exact score")
    scatter!(ax, yC[1, :], yC[2, :], color = :green, label = "Ludo Score")
    scatter!(ax, x⃗[1, :], x⃗[2, :], color = (:yellow, 0.1), label = "Samples")
    if k == 4
        axislegend(ax, position = :rt, orientation = :vertical)
    end
    xlims!(-2, 2)
    ylims!(-20, 20)
    display(fig)
end

##
gm_pdfs = []
for index_choice in ProgressBar(1:4)
    σ = σs[index_choice]
    δ(i, j) = i == j ? 1.0 : 0.0
    covy0 = [δ(i, j) * σ for i in 1:2, j in 1:2, k in 1:size(x⃗s[index_choice], 2)]
    gm = GaussianMixture([1/size(x⃗s[index_choice], 2) for i in eachindex(x⃗s[index_choice])], x⃗s[index_choice], covy0)
    extrema(x⃗s[2][:, 1])
    xs = range(-2, 2, length = 100)
    ys = range(-20, 20, length = 100)
    gm_pdf = [gm([x, y]) for x in xs, y in ProgressBar(ys)]
    push!(gm_pdfs, gm_pdf)
end
##
fig = Figure(resolution = (1200, 1200))
σs = [0.01, 0.05, 0.1, 0.5]
for (k, σ) in enumerate(σs)
    ii = (k-1)÷2 + 1
    jj = (k-1) % 2 + 1
    yC = yCs[k]
    gm_score = gm_scores[k]
    ax = Axis(fig[ii, jj]; xlabel = "x", ylabel = "Score", title = "Score comparison,  σ = $σ")
    xs = range(-2, 2, length = 100)
    ys = range(-20, 20, length = 100)
    heatmap!(ax, xs, ys, gm_pdfs[k], colormap = :grays, alpha = 0.8)
    contour!(ax, xs, ys, gm_pdfs[k], levels = 20, linewidth = 3, color = (:yellow, 0.05))
    lines!(ax, sort(x₀), gm_score[sortperm(x₀)], color = (:orange, 0.5), linewidth = 3, label = "GM score")
    lines!(ax, sort(x₀), exact_score[sortperm(x₀)], color = (:red, 0.5), linewidth = 3, label = "Exact score")
    scatter!(ax, yC[1, :], yC[2, :], color = :yellow, label = "KGM Score")
    # scatter!(ax, x⃗[1, :], x⃗[2, :], color = (:yellow, 0.1), label = "Samples")
    if k == 4
        axislegend(ax, position = :rt, orientation = :vertical)
    end
    xlims!(-2, 2)
    ylims!(-20, 20)
end
display(fig)
save("score_estimates.png", fig)
##

