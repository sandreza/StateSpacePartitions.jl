var documenterSearchIndex = {"docs":
[{"location":"function_index/#sec:function_index","page":"Function Index","title":"List of functions in StateSpacePartitions","text":"","category":"section"},{"location":"function_index/","page":"Function Index","title":"Function Index","text":"Modules = [ StateSpacePartitions.Trees, StateSpacePartitions]","category":"page"},{"location":"function_index/#StateSpacePartitions.Trees.Tree-Tuple{}","page":"Function Index","title":"StateSpacePartitions.Trees.Tree","text":"Tree(; structured = false, arguments = (; minimum_probability = 0.01))\n\nDescription\n\nA tree is a data structure that can be used to store partitions of a state space.\n\nKeyword Arguments\n\nstructured::Bool - Whether the tree is structured or not.\narguments::NamedTuple - The arguments for the tree.\n\nReturns\n\nTree - Either a BinaryTree or an UnstructuredTree.\n\n\n\n\n\n","category":"method"},{"location":"function_index/#StateSpacePartitions.Trees.determine_partition-Union{Tuple{S}, Tuple{Any, Tree{Val{false}, S}}} where S","page":"Function Index","title":"StateSpacePartitions.Trees.determine_partition","text":"determine_partition(trajectory, tree_type::Tree{Val{false}, S}; override = false) where S\n\nDecription\n\nThis function determines the partition of a trajectory into an unstructured tree. The tree structure is specified by the tree_type argument. The trajectory argument is a trajectory of states. The override keyword argument is a boolean indicating whether to override the truncation of the trajectory.\n\nArguments\n\ntrajectory: a trajectory of states\ntree_type: a Tree type\n\nKeyword Arguments\n\noverride: a boolean indicating whether to override the truncation of the trajectory\n\nReturns\n\nembedding: a Tree object\n\n\n\n\n\n","category":"method"},{"location":"function_index/#StateSpacePartitions.Trees.determine_partition-Union{Tuple{S}, Tuple{Any, Tree{Val{true}, S}}} where S","page":"Function Index","title":"StateSpacePartitions.Trees.determine_partition","text":"determine_partition(trajectory, tree_type::Tree{Val{true}, S}; override = false) where S\n\nDecription\n\nThis function determines the partitioning of the state space into a binary tree. The tree is determined by recursively splitting the state space into two parts. The splitting is done by k-means clustering. The number of levels of the tree is determined by the levels keyword argument. If the trajectory is too long, it is truncated to roughly 1000 points per level. The override keyword argument can be used to override this behavior.\n\nArguments\n\ntrajectory: a trajectory of states\ntree_type: a Tree type\n\nKeyword Arguments\n\noverride: a boolean indicating whether to override the truncation of the trajectory\n\nReturns\n\nembedding: a Tree object\n\n\n\n\n\n","category":"method"},{"location":"#StateSpacePartitions.jl","page":"Home","title":"StateSpacePartitions.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for StateSpacePartitions.jl","category":"page"}]
}
