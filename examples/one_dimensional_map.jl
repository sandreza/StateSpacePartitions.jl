# Potential well test
using StateSpacePartitions, ProgressBars, GLMakie, MarkovChainHammer, LinearAlgebra, Random
Random.seed!(1234)
function S(x)
    if x ≤ 1/3
        return 2x/(1-x)
    else
        return (1-x)/(2x)
    end
end

include("StatisticalModels.jl")


x = rand()
Nₜ = round(Int, 1e6)
xs = zeros(Nₜ)
xs[1] = x
for t in ProgressBar(1:Nₜ-1)
    xs[t+1] = S(xs[t])
end
ρ(y) = 2/((1 + y)^2)
fig = Figure()
ax = Axis(fig[1,1])
hist!(ax, xs, bins = 100, normalization = :pdf)
ys = range(0, 1, length = 10000)
lines!(ax, ys, ρ.(ys), color = (:red, 0.5), strokewidth = 10)

##
cells = round(Int, 1000 * 1.5)
ssp = StateSpacePartition(reshape(xs, (1, Nₜ)); cells)
δmodel = DeltaFunction(ssp, reshape(xs, (1, Nₜ)))
##
g(x) = (x[1] - 0.3862684534111812)^2
em = ensemble_mean(g, δmodel)
sum(ρ.(ys) .* g.(ys)) * (ys[2] - ys[1])
