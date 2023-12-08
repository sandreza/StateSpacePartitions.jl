using LinearSolve, LinearSolvePardiso
using SparseArrays, MarkovChainHammer, LinearAlgebra, ProgressBars, StateSpacePartitions
using IncompleteLU
using GLMakie

# include("sparse_chain_hammer.jl")
# include("sparse_ops.jl")
include("chaotic_systems.jl")
include("timestepping_utils.jl")
dt = 0.01 
iterations = 10^6
timeseries = zeros(3, iterations)
timeseries[:, 1] .= [14.0, 20.0, 27.0]
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(lorenz, timeseries[:, i-1], dt)
    timeseries[:, i] .= step.xⁿ⁺¹
end

##
levels = 11
tree_type = Tree(true, levels)
state_space_partitions = StateSpacePartition(timeseries; method = tree_type)
##
coarsen_level = 1
tmp = coarsen(state_space_partitions, coarsen_level)
pf = sparse_generator(state_space_partitions.partitions; dt = dt)'
coarse_pfo = Array(sparse_generator(tmp; dt = dt)')
from_end = 3
μ0, b = extract_coarse_guess(coarse_pfo, coarsen_level, 2^(levels - coarsen_level)-from_end)    

A = pf
x, λ = inverse_iteration(A, real.(b), real(μ0); tol = 1e-4, maxiter_eig = 10, maxiter_solve = 2)
λ
x
##
skip = 10
reduced_partitions = state_space_partitions.partitions[1:skip:end]
koopman_colors = [real(x[reduced_partitions[i]]) for i in eachindex(reduced_partitions)]
koopman_colors_guess = [real(b[reduced_partitions[i]]) for i in eachindex(reduced_partitions)]

fig = Figure(size = (927, 554))
ms= 5
a = 0.05
rotate_amount = (0, 11, 0)
ax = LScene(fig[1,2]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip:end], color=koopman_colors, colorrange = (-a, a), colormap=:balance, markersize=ms)
rotate_cam!(ax.scene, rotate_amount)
ax = LScene(fig[1,1]; show_axis=false)
a = 0.05
scatter!(ax, timeseries[:, 1:skip:end], color=koopman_colors_guess, colorrange = (-a, a), colormap=:balance, markersize=ms)
rotate_cam!(ax.scene, rotate_amount)
display(fig)
##
minimum_probability = 1e-4
tree_type = Tree(false, minimum_probability)
state_space_partitions = StateSpacePartition(timeseries; method = tree_type)
##
skip = 10
perron_frobenius_matrix = sparse_perron_frobenius(state_space_partitions.partitions; step = skip )
generator_matrix = sparse_generator(state_space_partitions.partitions; dt = dt)

##
pfo = generator_matrix' # generator_matrix'
target = -0.9
μ0 = target  
b = randn(size(pfo)[2]) # test b
A = copy(pfo)
x, λ = inverse_iteration(A, b, μ0; tol = 1e-10, maxiter_eig = 10, maxiter_solve = 2)
λ

##
pfo = perron_frobenius_matrix' # generator_matrix'
target = exp(λ * skip * dt) # 0.994
μ0 = target  
b = copy(x)
A = copy(pfo)
y, λ_pf = inverse_iteration(A, b, μ0; tol = 1e-10, maxiter_eig = 10, maxiter_solve = 2)
λ_pf
log(λ_pf) / (skip * dt)
##
skip_data = round(Int, 1/(100 * dt))
reduced_partitions = state_space_partitions.partitions[1:skip_data:end]
koopman_colors = [real(x[reduced_partitions[i]]) for i in eachindex(reduced_partitions)]
koopman_colors_pf = [real(y[reduced_partitions[i]]) for i in eachindex(reduced_partitions)]
# koopman_colors_pf_10 = copy(koopman_colors)

fig = Figure(size = (927, 554))
ms = 5
a = 100 * minimum_probability
rotate_amount = (0, 11, 0)
ax = LScene(fig[1,1]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip_data:end], color=koopman_colors, colorrange = (-a, a), colormap=:balance, markersize=ms)
rotate_cam!(ax.scene, rotate_amount)
ax = LScene(fig[1,2]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip_data:end], color=koopman_colors_pf, colorrange = (-a, a), colormap=:balance, markersize=ms)
rotate_cam!(ax.scene, rotate_amount)
display(fig)