@reexport module Trees 

using ParallelKMeans, LinearAlgebra

export Tree, determine_partition, unstructured_tree, UnstructuredTree, BinaryTree

struct Tree{E, P}
    structured::E 
    arguments::P 
end

Tree(; structured = false, arguments = (; minimum_probability = 0.01)) = Tree(structured, arguments)
Tree(bool::Bool, args::Any) = Tree(Val(bool), args)

include("unstructured.jl")
include("binary.jl")

end # module Tree