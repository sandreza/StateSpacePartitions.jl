using Enzyme
##
cells = round(Int, 1000 * 1.4)
ssp = StateSpacePartition(trajectory; cells)
##
Σmodel = GaussianMixture(ssp, trajectory)
# general_inflate!(Σmodel, cov(trajectory') * 0.01)
# isotropic_inflate!(Σmodel, 0.1)
# scaled_inflate!(Σmodel, 2.0)
δmodel = DeltaFunction(ssp)
## 
cov(Σmodel) - cov(δmodel) - average_covariance(Σmodel)
cov(trajectory')
scatter(rand(Σmodel, 1000))
samples_fixed_cell(Σmodel, 1, 1)


fig = Figure()
ax = Axis(fig[1, 1])
cell_number = 1
fixed_cell_samples_1 = samples_fixed_cell(Σmodel, cell_number, 1276)[1, :]
fixed_cell_samples_trajectory_1 = trajectory[:, ssp.partitions .== cell_number][1, :]
GLMakie.density!(ax, fixed_cell_samples_1, color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
hist!(ax, fixed_cell_samples_trajectory_1, color = (:blue, 0.5), bins = 100, normalization = :pdf)
cell_number = 2
fixed_cell_samples_2 = samples_fixed_cell(Σmodel, cell_number, 1276)[1, :]
fixed_cell_samples_trajectory_2 = trajectory[:, ssp.partitions .== cell_number][1, :]
GLMakie.density!(ax, fixed_cell_samples_2, color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
hist!(ax, fixed_cell_samples_trajectory_2, color = (:blue, 0.5), bins = 100, normalization = :pdf)
ax = Axis(fig[1, 2])
# combine 
hist!(ax, vcat(fixed_cell_samples_trajectory_1, fixed_cell_samples_trajectory_2), color = (:blue, 0.5), bins = 100, normalization = :pdf)
GLMakie.density!(ax, vcat(fixed_cell_samples_1, fixed_cell_samples_2), color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
display(fig)
##

fig = Figure() 
model_samples = rand(Σmodel, 100000)
for i in 1:3
    ax = Axis(fig[1, i])
    GLMakie.density!(ax, model_samples[i, :], color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
    hist!(ax, trajectory[i, :], color = (:blue, 0.5), bins = 100, normalization = :pdf)
end
display(fig)


## 
# Potential well test
function V(x)
    return (x[1]^2 - 1)^2/4 # + x[1] * 0.1
end

∇V(x) =  -gradient(Enzyme.Reverse, V, x)
∇V([2.0])
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
cells = round(Int, 1000 * 1.5)
ssp = StateSpacePartition(reshape(x₀, (1, Nₑ)); cells)
##
Σmodel = GaussianMixture(ssp, reshape(x₀, (1, Nₑ)))
scaled_inflate!(Σmodel, 4.0)
# general_inflate!(Σmodel, reshape([cov(x₀)], (1, 1)) * 0.04)

fig = Figure() 
ax = Axis(fig[1,1]) 
GLMakie.density!(ax, x₀[:], color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
lines!(ax, xs, normalized_density)
GLMakie.density!(ax, rand(Σmodel, 100000)[:], color = (:blue, 0.1), strokecolor = :blue,  strokewidth = 3)
display(fig)
##
score = ScoreModel(Σmodel)

model_score_values = [score([x])[1] for x in xs]
mollified_model_score_values = [score([x], 0.1)[1] for x in xs]
model_score_values_on_data = [score([x])[1] for x in x₀[1:100:end]]
exact_score_values = [∇V([x])[1] for x in xs]

fig = Figure() 
ax = Axis(fig[1,1])
xlims!(ax, -1.2, 1.2)
ylims!(ax, -2, 2)
lines!(ax, xs, exact_score_values, color = :red)
lines!(ax, xs, mollified_model_score_values, color = :green)
scatter!(ax, xs, model_score_values, color = :blue)

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
GLMakie.density!(ax, fixed_cell_samples_2, color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
hist!(ax, fixed_cell_samples_trajectory_2, color = (:blue, 0.5), bins = 100, normalization = :pdf)
ax = Axis(fig[1, 2])
# combine 
hist!(ax, vcat(fixed_cell_samples_trajectory_1, fixed_cell_samples_trajectory_2), color = (:blue, 0.5), bins = 100, normalization = :pdf)
GLMakie.density!(ax, vcat(fixed_cell_samples_1, fixed_cell_samples_2), color = (:red, 0.1), strokecolor = :red,  strokewidth = 3)
display(fig)