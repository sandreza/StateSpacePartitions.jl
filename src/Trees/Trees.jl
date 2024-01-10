@reexport module Trees 

using ParallelKMeans, LinearAlgebra
using StateSpacePartitions.Architectures

export Tree, determine_partition, unstructured_tree, UnstructuredTree, BinaryTree

struct Tree{E, P}
    structured::E 
    arguments::P 
end

"""
    Tree(; structured = false, arguments = (; minimum_probability = 0.01))

# Description 

A tree is a data structure that can be used to store partitions of a state space.

# Keyword Arguments

* `structured::Bool` - Whether the tree is structured or not.
* `arguments::NamedTuple` - The arguments for the tree.

# Returns 

* `Tree` - Either a BinaryTree or an UnstructuredTree.

"""
Tree(; structured = false, arguments = (; minimum_probability = 0.01)) = Tree(structured, arguments)
Tree(bool::Bool, args::Any) = Tree(Val(bool), args)

include("unstructured.jl")
include("binary.jl")

end # module Tree