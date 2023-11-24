import StateSpacePartitions.Trees: unstructured_tree, UnstructuredTree, determine_partition, Tree

@testset "Trees" begin
    @test typeof(Tree(false, 1)) == Tree{Val{false}, Int64}
end
