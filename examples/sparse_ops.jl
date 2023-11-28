function inverse_iteration(A, x0, μ0; tol = 1e-4, maxiter_eig = 10, maxiter_solve = 2)
    prob = LinearProblem(A, x0)
    @info "computing incomplete LU factorization"
    pl  = ilu(A, τ = 0.1)
    x = x0 / norm(x0)
    λ = x' * A * x
    @info "looping through iterative eigensolver"
    for i in ProgressBar(1:maxiter_eig)
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

# no need for binary coarsen function
function coarsen(ssp::StateSpacePartition{S, A}, levels) where {S <: BinaryTree, A} 
    partitions = ssp.partitions
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
