using StateSpacePartitions, ProgressBars, Random, GLMakie 
using GraphMakie, NetworkLayout, Graphs, Printf, SparseArrays
using MarkovChainHammer, LinearAlgebra, LinearSolve, Statistics
using IncompleteLU
include("chaotic_systems.jl")
include("sparse_ops.jl")
include("timestepping_utils.jl")
include("visualization_utils.jl")
include("coarsening_utils.jl")
include("sparse_chain_hammer.jl")
Random.seed!(1234)

dt = 0.001 
Tfinal = 10^3
iterations = round(Int, Tfinal/dt)

timeseries = zeros(3, iterations)
timeseries[:, 1] .= [14.0, 20.0, 27.0]
step = RungeKutta4(3)
for i in ProgressBar(2:iterations)
    step(lorenz, timeseries[:, i-1], dt)
    timeseries[:, i] .= step.xⁿ⁺¹
end
##
# dt/10 is roughly p_min
p_min = dt/10
@info "computing embedding"
Nmax = 100 * round(Int, 1/ p_min)
skip = maximum([round(Int, size(timeseries)[2] / Nmax), 1])
F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(timeseries[:, 1:skip:end], p_min; threshold = 2)
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI);
G = SimpleDiGraph(adj)
embedding = UnstructuredTree(P4, C, P3)
partitions = zeros(Int64, size(timeseries)[2])
@info "computing partition timeseries"
for i in ProgressBar(eachindex(partitions))
    @inbounds partitions[i] = embedding(timeseries[:, i])
end
##
graph_edges = PI 
probability_minimum = maximum([0.001, p_min * 10])
parent_to_children = P3 
global_to_local = P4
@info "computing coarse partitioning function"
local_to_local = unstructured_coarsen_edges(graph_edges, probability_minimum, parent_to_children, G, global_to_local)
coarse_partitions = zeros(Int64, size(timeseries)[2])
@info "computing coarse partition timeseries"
for i in ProgressBar(eachindex(coarse_partitions))
    @inbounds coarse_partitions[i] = local_to_local[partitions[i]]
end
##
rotate_amount = (0, 11, 0)
fig = Figure()
ax = LScene(fig[1,1]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip:end], color=partitions[1:skip:end], colormap=:glasbey_hv_n256, markersize=10)
rotate_cam!(ax.scene, rotate_amount)
ax = LScene(fig[1,2]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip:end], color=coarse_partitions[1:skip:end], colormap=:glasbey_hv_n256, markersize=10)
rotate_cam!(ax.scene, rotate_amount)
display(fig)
##
Q = generator(coarse_partitions; dt = dt)
Λ, W = eigen(Q')
coarse_koopman = real.(W[:, end-3])
coarse_λ = real(Λ[end-3])
##
refine_koopman = zeros(eltype(coarse_koopman), maximum(partitions))
for i in eachindex(refine_koopman)
    refine_koopman[i] = coarse_koopman[local_to_local[i]]
end
##
sQ = sparse_generator(partitions; dt = dt)
avg_diag = tr(sQ) / size(sQ)[1]
1/(avg_diag * dt)

sQᵀ = sQ'
μ0 = coarse_λ / 2
b = copy(refine_koopman)
A = copy(sQᵀ)
for i in ProgressBar(eachindex(b))
    A[i, i] = A[i, i] - μ0 
end
koopman, λ = inverse_iteration(A, b, μ0; tol = 1e-10, maxiter_eig = 100, maxiter_solve = 5)
λ
##
skip_data = round(Int, 1/(100 * dt))
reduced_coarse_partitions = coarse_partitions[1:skip_data:end]
reduced_partitions = partitions[1:skip_data:end]
coarse_koopman_colors = [real(coarse_koopman[reduced_coarse_partitions[i]]) for i in eachindex(reduced_coarse_partitions)]
koopman_colors_refine = [real(refine_koopman[reduced_partitions[i]]) for i in eachindex(reduced_partitions)]
koopman_colors = [real(koopman[reduced_partitions[i]]) for i in eachindex(reduced_partitions)]
##
#=
fig = Figure()
inds = 1:100:3*100000
koopman_timeseries = [-real(koopman[partitions[i]]) for i in inds]
a = quantile(koopman_timeseries, 0.9)
for i in 1:3
    ax = Axis(fig[i, 1])
    scatter!(ax, timeseries[i, inds], color=koopman_timeseries, colormap=:balance, colorrange = (-a, a))
end
ax = Axis(fig[4, 1])
scatter!(ax, koopman_timeseries)
=#
##
fig = Figure(size = (927, 554))
ms = 5
a = maximum(coarse_koopman_colors) 
rotate_amount = (0, 11, 0)
ax = LScene(fig[1,1]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip_data:end], color=coarse_koopman_colors, colorrange = (-a, a), colormap=:balance, markersize=ms)
rotate_cam!(ax.scene, rotate_amount)
ax = LScene(fig[1,2]; show_axis=false)
scatter!(ax, timeseries[:, 1:skip_data:end], color=koopman_colors_refine, colorrange = (-a, a), colormap=:balance, markersize=ms)
rotate_cam!(ax.scene, rotate_amount)
ax = LScene(fig[1,3]; show_axis=false)
a = quantile(koopman_colors, 0.9) 
scatter!(ax, timeseries[:, 1:skip_data:end], color=koopman_colors, colorrange = (-a, a), colormap=:balance, markersize=ms)
rotate_cam!(ax.scene, rotate_amount)
display(fig)

##
function refine_koopman_mode(local_to_local_coarse, local_to_local_fine, coarse_koopman)
    koopman = zeros(eltype(coarse_koopman), maximum(values(local_to_local_fine)))
    reverse_dictionary = Dict(value => key for (key, value) in local_to_local_fine)
    for i in eachindex(koopman)
        ii = reverse_dictionary[i]
        coarse_i = local_to_local_coarse[ii]
        koopman[i] = coarse_koopman[coarse_i]
    end
    return koopman
end
##
probability_ranges = reverse((10 .^(range(log10.(2 * p_min), -2, step = 0.19))))
local_to_locals = []
coarsened_partitions = Vector{Int64}[]
for p_min in ProgressBar(probability_ranges)
    probability_minimum = p_min
    graph_edges = PI 
    parent_to_children = P3 
    global_to_local = P4
    local_to_local = unstructured_coarsen_edges(graph_edges, probability_minimum, parent_to_children, G, global_to_local)
    push!(local_to_locals, local_to_local)
    coarse_partitions = zeros(Int64, size(timeseries)[2])
    @info "computing coarse partition timeseries"
    for i in ProgressBar(eachindex(coarse_partitions))
        @inbounds coarse_partitions[i] = local_to_local[partitions[i]]
    end
    push!(coarsened_partitions, coarse_partitions)
end
##
Q = generator(coarsened_partitions[1]; dt = dt)
Λ, W = eigen(Q')
coarse_koopman = real.(W[:,end-3])#  real.(W[:, end-3])
coarse_λ = real.(Λ[end-3]) # real(Λ[end-3])

λs = typeof(coarse_λ)[]
koopmans = Vector{typeof(coarse_λ)}[]
push!(λs, coarse_λ)
push!(koopmans, coarse_koopman)
for i in ProgressBar(2:length(coarsened_partitions))
    sQ = sparse_generator(coarsened_partitions[i]; dt = dt)
    sQᵀ = sQ'
    μ0 = coarse_λ
    local_to_local_coarse = local_to_locals[i-1]
    local_to_local_fine = local_to_locals[i]
    b = refine_koopman_mode(local_to_local_coarse, local_to_local_fine, coarse_koopman)
    A = copy(typeof(coarse_λ).(sQᵀ))
    for i in ProgressBar(eachindex(b))
        A[i, i] = A[i, i] - μ0 
    end
    koopman, λ = inverse_iteration(A, b, μ0; tol = 1e-10, maxiter_eig = 10, maxiter_solve = 2)
    push!(λs, λ)
    push!(koopmans, koopman)
    coarse_λ = copy(λ)
    coarse_koopman = copy(koopman)
end

##
fig = Figure()
N = minimum([4, floor(Int,sqrt(length(koopmans)))])
shift = maximum([length(koopmans) - N^2, 0])
ichoice = range(1, length(koopmans), length = N^2)
for i in 1:N^2
    ii = (i-1)÷N
    jj = (i-1)%N
    shift_i = round(Int, ichoice[i]) # i + shift
    ax = LScene(fig[ii+1,jj+1]; show_axis=false)
    println((ii+1,jj+1), " => ", maximum(coarsened_partitions[shift_i]))
    koopman_colors = [koopmans[shift_i][coarse_partition] for coarse_partition in coarsened_partitions[shift_i]][1:skip_data:end]
    if eltype(koopman_colors) <: ComplexF64
        a = quantile(abs.(koopman_colors), 0.9)
        b = quantile(abs.(koopman_colors), 0.1)
        koopman_colors = abs.(koopman_colors)
        colormap = :afmhot
        colorrange = (b, a)
        #=
        a = quantile(imag.(koopman_colors), 0.9)
        b = quantile(imag.(koopman_colors), 0.1)
        koopman_colors = imag.(koopman_colors)
        colormap = :balance
        colorrange = (-a, a)
        =#
    else
        a = quantile(koopman_colors, 0.95)
        colormap = :balance
        colorrange = (-a, a)
    end
    scatter!(ax, timeseries[:, 1:skip_data:end], color=koopman_colors, colorrange = colorrange, colormap=colormap, markersize=ms)
    rotate_cam!(ax.scene, rotate_amount)
end
display(fig)
##
fig = Figure()
partition_numbers = maximum.(coarsened_partitions)
ls = 40
ax = Axis(fig[1,1]; ylabel = "eigenvalue", xlabel = "log10(partitions)", xlabelsize = ls, ylabelsize = ls, xticklabelsize = ls, yticklabelsize = ls)
scatter!(ax, log10.(partition_numbers), λs)
display(fig)
##
scatter(partition_numbers, real.(λs))
# λ =  a + c / partition_number
ind1 = length(partition_numbers)
ind2 = length(partition_numbers) - 1
a, c = [1 1/partition_numbers[ind1]; 1 1/partition_numbers[ind2]] \ [λs[ind1]; λs[ind2]]
m = 2
mat= zeros(m,m)
for i in 1:m 
    for j in 1:m
        mat[i,j] = 1/partition_numbers[end-i+1]^(j-1)
    end
end
vec = mat \ reverse(λs[end-m+1:end])
a .+ c ./ (partition_numbers)
partition_numbers, λs
# more sophisticated fit would be λ + O(dx) + O(dx^2)

##
shift_i = round(Int, ichoice[end])
koopman_colors = [koopmans[shift_i][coarse_partition] for coarse_partition in coarsened_partitions[shift_i]][1:skip_data:end]

##
lines(real.(koopman_colors)[1+1000:16000+1000], imag.(koopman_colors)[1+1000:16000+1000])
##
lines(real.(koopman_colors)[1:1000])