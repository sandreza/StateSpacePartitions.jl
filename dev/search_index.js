var documenterSearchIndex = {"docs":
[{"location":"function_index/#sec:function_index","page":"Function Index","title":"List of functions in StateSpacePartitions","text":"","category":"section"},{"location":"function_index/","page":"Function Index","title":"Function Index","text":"Modules = [ StateSpacePartitions.Trees, StateSpacePartitions]","category":"page"},{"location":"function_index/#StateSpacePartitions.Trees.Tree-Tuple{}","page":"Function Index","title":"StateSpacePartitions.Trees.Tree","text":"Tree(; structured = false, arguments = (; minimum_probability = 0.01))\n\nDescription\n\nA tree is a data structure that can be used to store partitions of a state space.\n\nKeyword Arguments\n\nstructured::Bool - Whether the tree is structured or not.\narguments::NamedTuple - The arguments for the tree.\n\nReturns\n\nTree - Either a BinaryTree or an UnstructuredTree.\n\n\n\n\n\n","category":"method"},{"location":"function_index/#StateSpacePartitions.inverse_iteration-Tuple{Any, Any, Any}","page":"Function Index","title":"StateSpacePartitions.inverse_iteration","text":"inverse_iteration(A, x₀, μ₀; tol = 1e-4, maxiter_eig = 10, maxiter_solve = 2, τ = 0.1)\n\nDescription\n\nComputes the eigenpair (λ, x) of the matrix A corresponding to initial eigenguess (μ₀, x₀) using inverse iteration.\n\nArguments\n\nA::Matrix - The matrix to compute the eigenpair of.\nx₀::Vector - The initial guess for the eigenvector.\nμ₀::Real - The initial guess for the eigenvalue.\n\nKeyword Arguments\n\ntol::Real - The Cauchy-criteria tolerance for the iterative eigensolver.\nmaxiter_eig::Int - The maximum number of iterations for the iterative eigensolver.\nmaxiter_solve::Int - The maximum number of iterations for the iterative linear solver.\nτ::Real - The drop tolerance for the incomplete LU factorization.\n\nReturns\n\nx::Vector - The eigenvector.\nλ::Real - The eigenvalue.\n\n\n\n\n\n","category":"method"},{"location":"#StateSpacePartitions.jl","page":"Home","title":"StateSpacePartitions.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for StateSpacePartitions.jl","category":"page"}]
}
