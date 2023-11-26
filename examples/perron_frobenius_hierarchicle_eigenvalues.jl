using LinearSolve, LinearSolvePardiso
using SparseArrays, MarkovChainHammer, LinearAlgebra, ProgressBars, StateSpacePartitions
using IncompleteLU

function inverse_iteration(A, x0, μ0; tol = 1e-6, maxiter_eig = 2, maxiter_solve = 2)
    prob = LinearProblem(A, x0)
    pl  = ilu(A, τ = 0.1)
    x = x0 / norm(x0)
    λ = x' * A * x
    for i in 1:maxiter_eig
        x = solve(prob, KrylovJL_GMRES(), Pl = pl, max_iters = maxiter_solve).u
        x = x / norm(x)
        λ_old = λ
        λ = x' * A * x
        if abs(λ - λ_old) < tol
            break
        end
        prob = LinearProblem(A, x)
    end
    return x, λ + μ0
end

coarsen(ssp::StateSpacePartition{S, A}, levels) where {S <: BinaryTree, A} = binary_coarsen(ssp.partitions, levels)
function binary_coarsen(partitions::Vector{Int64}, levels)
    return (partitions .- 1) .÷ 2^levels .+ 1
end
function extract_coarse_guess(coarse_pfo, levels, index)
    ll, vv = eigen(coarse_pfo)
    guess = zeros(ComplexF64 ,size(coarse_pfo)[1] * 2^levels)
    for i in eachindex(guess)
        guess[i] = vv[(i-1)÷(2^levels) + 1, index]
    end
    return ll[index], guess / norm(guess)
end


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
operator = generator # perron_frobenius
coarse_pfo = operator(tmp)'
from_end = 3
μ0, b = extract_coarse_guess(coarse_pfo, coarsen_level, 2^(levels - coarsen_level)-from_end)    
# main idea will be to start from base.eigen calculation with 256x256 mat and then work up to 2^levels x 2^levels

pfo = operator(state_space_partitions.partitions)'
pf = sparse(operator(state_space_partitions.partitions)')
# target = 0.9312661899815546
# μ0 = target  - 1e-3
# b = ones(size(A)[2]) # test b
A = pf - μ0*I  

x, λ = inverse_iteration(A, b, μ0; tol = 1e-16, maxiter_eig = 10, maxiter_solve = 2)
λ
x
# x' * A * x + μ0
##
skip = 10
reduced_partitions = state_space_partitions.partitions[1:skip:end]
koopman_colors = [real(x[reduced_partitions[i]]) for i in eachindex(reduced_partitions)]
koopman_colors_guess = [real(b[reduced_partitions[i]]) for i in eachindex(reduced_partitions)]

using GLMakie
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
#=
ll, vv = eigen(pfo)
ll[end-from_end] - λ
vv[:, end-from_end] * sign(vv[1, end-from_end]) - x * sign(x[1])
scaled_entropy(real.(vv[:, end] / sum(vv[:, end])))
##
koopman_colors = [real(x[state_space_partitions.partitions[i]]) for i in eachindex(state_space_partitions.partitions)]
koopman_colors2 = [real(vv[state_space_partitions.partitions[i], end-from_end]) for i in eachindex(state_space_partitions.partitions)]
using GLMakie
fig = Figure()
skip = 10
a = -0.05
b = -a
ax = LScene(fig[1,1]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip:end], color=koopman_colors2[1:skip:end], colorrange = (a, b), colormap=:balance, markersize=10)
ax = LScene(fig[1,2]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip:end], color=koopman_colors[1:skip:end], colorrange = (a, b), colormap=:balance, markersize=10)
display(fig)
=#