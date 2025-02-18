ps = [Point3f(x, y, z) for x in -5:2:5 for y in -5:2:5 for z in -5:2:5]
ns = map(p -> 0.1 * Vec3f(p[2], p[3], p[1]), ps)
lengths = norm.(ns)
arrows(
    ps, ns, fxaa=true, # turn on anti-aliasing
    color=lengths,
    linewidth = 0.1, arrowsize = Vec3f(0.3, 0.3, 0.4),
    align = :center
)
##
trj = trajectory[:, 1:100:end]
lorenz_score = ScoreModel(Σmodel)
score_on_trajectory = [Vec3f(lorenz_score(x⃗)) for x⃗ in eachcol(trj)] 
ps = [Point3f(x, y, z) for (x, y, z) in eachcol(trj)]

arrows(ps, score_on_trajectory, fxaa=true, color = :blue, linewidth = 0.1, arrowsize = 0.3, align = :center)
# scatter!(trj)
##
cells = round(Int, 100 * 1.4)
ssp = StateSpacePartition(trajectory; cells)
##
Σmodel = GaussianMixture(ssp, trajectory)
# isotropic_inflate!(Σmodel, 10.0)
scaled_inflate!(Σmodel, 4.0)
##
trj = trajectory[:, 1:1000:end]
mat = zeros(3,3)
lorenz_score = ScoreModel(Σmodel)
N = length(eachcol(trj))
gaussian_model = GaussianMixture([1.0], mean(trj, dims = 2), reshape(cov(trj'), 3, 3, 1))
nlist = []
for (i, x⃗) in ProgressBar(enumerate(eachcol((Σmodel.means))))
    # NN = randn(3) * 0.0
    # push!(nlist, NN)
    z⃗ = x⃗ # + NN
    mat .+= -z⃗ * lorenz_score(z⃗)' * Σmodel.weights[i]
end
mat
##
mat = zeros(3,3)
gaussian_score = ScoreModel(gaussian_model)
lorenz_score = ScoreModel(Σmodel)
samples = rand(gaussian_model, 100000)
scoreval = [gaussian_score(x⃗) for x⃗ in ProgressBar(eachcol(samples))]
for i in ProgressBar(eachindex(scoreval))
    mat .+= -samples[:, i] * scoreval[i]'
end

mat = zeros(3,3)
C = cov(trj')
μ = mean(trj, dims = 2)[:]
C⁻¹ = inv(C)
for x in eachcol(trj[:, 1:10:end])
    mat .+= x * (C⁻¹* (x -  μ))'
end