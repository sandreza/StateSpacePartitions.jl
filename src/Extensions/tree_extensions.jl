# Binary Tree 
function coarsen(ssp::StateSpacePartition{S, A}, levels) where {S <: BinaryTree, A} 
    partitions = ssp.partitions
    return (partitions .- 1) .÷ 2^levels .+ 1
end