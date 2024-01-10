module Architectures

using KernelAbstraction
using KernelAbstraction: CPU
using CUDA
using CUDA: CUDABackend

import Base

const GPU = CUDABackend

convert(::CPU, array::CuArray) = Array(array)
convert(::GPU, array::AbstractArray) = CuArray(array)
convert(::CPU, array::AbstractArray) = array
convert(::GPU, array::CuArray) = array

"""
    ChunkedArray{A, B, C, I}(architecture, array, chunked_array, current_range)

A struct representing a chunked array where the chunk lives on a different architecture than the array. 
When indexed into, the chunk on the architecture and the `current_range` are updated.

# Fields
- `architecture::A`: The architecture on which the chunk of the array is to be processed. 
- `array::B`: The actual array data.
- `chunked_array::C`: The chunk of the array that resides on the specified architecture.
- `current_range::I`: The range of indices that currently represent the chunk on the architecture.

# Example
```julia
architecture = CPU()
array = [1, 2, 3, 4, 5]
chunked_array = [1, 2]
current_range = 1:2
chunked_array = ChunkedArray(architecture, array, chunked_array, current_range)
```
"""
struct ChunkedArray{A, B, C, I}
    architecture :: A
    array :: B
    chunk :: C
    current_range :: I
end

function ChunkedArray(array, architecture = CPU(); chunk_size = length(array))
    chunk = convert(architecture, array[1:chunk_size])
    current_range = 1:chunk_size
    return ChunkedArray(architecture, array, chunk, current_range)
end

function update_array!(chunked_array::ChunkedArray, n::Int)
    if n ∈ chunked_array.current_range
        return nothing
    end

    arch  = chunked_array.architecture
    chunk = chunked_array.chunka
    array = chunked_array.array
    range = chunked_array.current_range
    Nt = length(array)
    Ni = length(chunk)
    if n == 1
        range .= 1:Ni
    elseif n > Nt - Ni
        range .= Nt-Ni+1:Nt
    else
    range .= n-1:n+Ni-2
    end
    
    chuck .= convert(arch, array[range])
    
    return nothing
end

# Chunk the kernel into smaller sizes
function launch_chunked_kernel!(arch, workgroup, worksize, _kernel!, args)
    chunked_arrays = findall(x -> x isa ChunkedArray, args)
    
    if isempty(chunked_arrays)
        loop! = _kernel!(arch, workgroup, worksize)
        loop!(args...)
        return nothing
    end    

    chunked_worksize = length(chuncked_arrays[1])
    
    M = worksize ÷ chunked_worksize
    loop! = _kernel!(arch, workgroup, chunked_worksize)

    for m in 1:M
        idx = m * chunked_worksize - 1
        
        # Updating the chunk!
        for chunked_array in chunked_arrays
            update_array!(chunked_array, idx)
        end

        loop!(args...)
    end

    return nothing
end

end