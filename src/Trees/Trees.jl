@reexport module Trees 

using ParallelKMeans, LinearAlgebra

export Tree, determine_partition, unstructured_tree, UnstructuredTree, BinaryTree

struct Tree{E, P}
    structured::E 
    arguments::P 
end

Tree() = Tree(false, (; minimum_probability = 0.01))
Tree(bool::Bool, args::Any) = Tree(Val(bool), args)

include("unstructured.jl")
include("binary.jl")

end # module Tree