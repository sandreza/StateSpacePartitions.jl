# Potential well test
function V(x)
    return (x[1]^2 - 1)^2 # + x[1] * 0.1
end

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
cells = round(Int, 4 * 1.5)
ssp = StateSpacePartition(reshape(x₀, (1, Nₑ)); cells)
##
Σmodel = GaussianMixture(ssp, reshape(x₀, (1, Nₑ)))
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
