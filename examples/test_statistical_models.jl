using Enzyme
##
cells = round(Int, 100 * 1.4)
ssp = StateSpacePartition(trajectory; cells)
##
Σmodel = GaussianMixture(ssp, trajectory)
scaled_inflate!(Σmodel, 2)
# isotropic_inflate!(Σmodel, 1.0)
# covmodel = cov(Σmodel)
# determinant_inflate!(Σmodel)
# covmodel = cov(Σmodel)
# scaled_inflate!(Σmodel, 2)
# covmodel = cov(Σmodel)
# general_inflate!(Σmodel, cov(trajectory') * 0.01)
# isotropic_inflate!(Σmodel, 0.1)
# scaled_inflate!(Σmodel, 2.0)
δmodel = DeltaFunction(ssp, trajectory)
## 
probabilities = [Σmodel(x⃗) for x⃗ in eachcol(trajectory[:, 1:100:end])] 
##
cmap = :thermal
n = size(trajectory[:, 1:100:end], 2)
alphas = probabilities / maximum(probabilities) * 0.1
cmap_alpha = resample_cmap(cmap, n; alpha = alphas)
points3d = [Point3f(x⃗[1], x⃗[2], x⃗[3]) for x⃗ in eachcol(trajectory[:, 1:100:end])];
meshscatter(vec(points3d); color = vec(probabilities), marker = Rect3f(Vec3f(-10), Vec3f(20)), colormap = cmap_alpha)
##
xrange = range(extrema(trajectory[1, :])..., length = 80)
yrange = range(extrema(trajectory[2, :])..., length = 80)
zrange = range(extrema(trajectory[3, :])..., length = 80)
isotropic_sample_probability = [Σmodel([x, y, z]) for x in xrange, y in yrange, z in zrange]

scatter(xrange, sum(isotropic_sample_probability, dims = (2, 3))[:])
##
cmap = :thermal # :linear_kryw_0_100_c71_n256
cmapa = reverse(RGBAf.(to_colormap(cmap)))
cmap = vcat(fill(RGBAf(0, 0, 0, 0), 1), cmapa[1:256])
volume(isotropic_sample_probability, algorithm = :absorption, absorption=5.0f0, colormap=cmap, transparency=true)
##
maxprob = maximum(probabilities)
colors = [(:red, probability/maxprob ) for probability in probabilities]
##
sortedxs = sort(trajectory[1, 1:100:end])
perxs = sortperm(trajectory[1, 1:100:end])
scatter(trajectory[:, 1:100:end], color = colors)
lines(sortedxs, probabilities[perxs])
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
fig = Figure() 
for i in 1:3
    for j in 1:3
        ax = Axis(fig[i, j])
        if i == j 
            hist!(ax, trajectory[i, :], bins = 30, normalization = :pdf, color = :red)
            density!(ax, model_samples[i, :][:], color = (:blue, 0.5))
        elseif i > j 
            scatter!(ax, model_samples[j, :], model_samples[i, :], color = :blue, markersize = 1.0)
        else
            scatter!(ax, trajectory[i, :], trajectory[j, :], color = :red, markersize = 1.0)
        end
    end
end
display(fig)


##
