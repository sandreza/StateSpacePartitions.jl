using Documenter
using StateSpacePartitions

makedocs(
    sitename="State Space Partitions",
    format=Documenter.HTML(collapselevel=1),
    pages=[
        "Home" => "index.md",
        "Function Index" => "function_index.md",
    ],
    modules=[StateSpacePartitions]
)

deploydocs(repo="github.com/sandreza/StateSpacePartitions.jl.git")