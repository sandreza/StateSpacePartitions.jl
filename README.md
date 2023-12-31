<!-- Title -->
<h1 align="center">
  StateSpacePartitions.jl
</h1>

A toolkit for analyzing dynamical systems through operator based approaches. See the [documentation](https://sandreza.github.io/StateSpacePartitions.jl/dev/) for examples.

 <a href="https://mit-license.org">
    <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
  </a>
 <a href="https://sandreza.github.io/StateSpacePartitions.jl/dev">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-stable%20release-red?style=flat-square">
  </a>
 <a href="https://github.com/SciML/ColPrac">
    <img alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square">

## Contents
* [Tools](#tools)
* [Installation instructions](#installation-instructions)
* [Contributing](#contributing)

## Tools

By default StateSpacePartitions exports no functions and has modules that
1. Construct [transfer operators](https://en.wikipedia.org/wiki/Transfer_operator) from data

See the [examples](https://github.com/sandreza/StateSpacePartitions.jl/tree/main/examples) for inspiration on how the utilities can be used.
## Installation instructions

StateSpacePartitions is a ***unregistered*** Julia package that requires Julia 1.8+. To install it,

1. [Download Julia](https://julialang.org/downloads/).
2. Launch Julia and type 
```julia
julia> using Pkg

julia> Pkg.add("https://github.com/sandreza/StateSpacePartitions.jl.git")
```

## Contributing 

We follow [Julia conventions](https://docs.julialang.org/en/v1/manual/style-guide/) and recommend reading through [ColPrac](https://docs.sciml.ai/ColPrac/stable/) as a standard guide for contributing to Julia software. New issues and pull requests are welcome!