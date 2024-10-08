using LinearAlgebra, LinearSolve, IncompleteLU
export inverse_iteration

"""
    inverse_iteration(A, x₀, μ₀; tol = 1e-4, maxiter_eig = 10, maxiter_solve = 2, τ = 0.1)

# Description 

Computes the eigenpair (λ, x) of the matrix A corresponding to initial eigenguess (μ₀, x₀) using inverse iteration.

# Arguments

* `A::Matrix` - The matrix to compute the eigenpair of.
* `x₀::Vector` - The initial guess for the eigenvector.
* `μ₀::Real` - The initial guess for the eigenvalue.

# Keyword Arguments
* `tol::Real` - The Cauchy-criteria tolerance for the iterative eigensolver.
* `maxiter_eig::Int` - The maximum number of iterations for the iterative eigensolver.
* `maxiter_solve::Int` - The maximum number of iterations for the iterative linear solver.
* `τ::Real` - The drop tolerance for the incomplete LU factorization.

# Returns

* `x::Vector` - The eigenvector.
* `λ::Real` - The eigenvalue.
"""
function inverse_iteration(A, x₀, μ₀; tol = 1e-4, maxiter_eig = 10, maxiter_solve = 2, τ = 0.1)
    B = copy(A)
    @info "Adjusting Diagonal"
    for i in ProgressBar(eachindex(x₀))
        B[i, i] = B[i, i] - μ₀ 
    end
    linear_problem = LinearProblem(B, x₀)
    @info "computing incomplete LU factorization"
    incomplete_lu_factorization  = ilu(B, τ = τ)
    x = x₀ / norm(x₀)
    λ = x' * B * x
    @info "looping through iterative eigensolver"
    for i in ProgressBar(1:maxiter_eig)
        x = solve(linear_problem, KrylovJL_GMRES(), Pl = incomplete_lu_factorization , max_iters = maxiter_solve).u
        x = x / norm(x)
        λ_old = λ
        λ = x' * B * x
        if abs(λ - λ_old) < tol
            break
        end
        linear_problem = LinearProblem(B, x)
    end
    return x, λ + μ₀
end