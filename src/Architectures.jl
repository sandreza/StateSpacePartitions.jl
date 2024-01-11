module Architectures

export CPU, GPU, ChunkedArray, update_chunk!, update_array!
export launch_chunked_kernel!

using ProgressBars
using KernelAbstractions
using KernelAbstractions: CPU
using CUDA
using CUDA: CUDABackend

import Base

const GPU = CUDABackend
const AA = AbstractArray
const CA = CuArray

convert(::CPU, array::CA) = Array(array)
convert(::GPU, array::AA) = CuArray(array)
convert(::CPU, array::AA) = array
convert(::GPU, array::CA) = array

vector_type(::CPU) = Vector
vector_type(::GPU) = CuVector

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
struct ChunkedArray{N, A, C, I}
    array :: A
    chunk :: C
    current_range :: I
    ChunkedArray{N}(arr::A, chunk::C, range::I) where {N, A, C, I} = new{N, A, C, I}(arr, chunk, range)
end

function ChunkedArray(array::AbstractArray{T, N}, architecture = CPU(); chunk_size = size(array, N)) where {T, N}
    chunk = convert(architecture, getlastdim(array, 1:chunk_size))
    current_range = collect(1:chunk_size)
    return ChunkedArray{N}(array, chunk, current_range)
end

getlastdim(a::AbstractArray{T, 1}, range) where T = @inbounds view(a, range)

function getlastdim(a::AbstractArray{T, N}, range) where {T, N}
    indices = Tuple(Colon() for _ in 1:N-1)
    return @inbounds view(a, indices..., range)
end

total_length(chunked_array::ChunkedArray{N}) where N = size(chunked_array.array, N)
Base.length(chunked_array::ChunkedArray) = length(chunked_array.current_range)

architecture(::AbstractArray) = CPU()
architecture(::CuArray) = GPU()
architecture(chunked::ChunkedArray) = architecture(chunked.chunk)

function update_chunk!(chunked_array::ChunkedArray, n::Int)
    if n ∈ chunked_array.current_range
        return nothing
    end

    arch  = architecture(chunked_array)
    chunk = chunked_array.chunk
    array = chunked_array.array
    range = chunked_array.current_range
    Nt = total_length(chunked_array)
    Ni = length(chunked_array)
    if n == 1
        range .= 1:Ni
    elseif n > Nt - Ni
        range .= Nt-Ni+1:Nt
    else
        range .= n-1:n+Ni-2
    end
    
    chunk .= convert(arch, getlastdim(array, range))
    
    return nothing
end

function update_array!(chunked_array)
    chunk = chunked_array.chunk
    array = chunked_array.array
    range = chunked_array.current_range

    getlastdim(array, range) .= convert(CPU(), chunk)

    return nothing
end

# Fallback
getchunk(a) = a
getchunk(c::ChunkedArray) = c.chunk

# Chunk the kernel into smaller sizes
function launch_chunked_kernel!(arch, workgroup, worksize, _kernel!, args)
    chunked_arrays = filter(x -> x isa ChunkedArray, args)
    
    if isempty(chunked_arrays)
        loop! = _kernel!(arch, workgroup, worksize)
        loop!(args...)
        return nothing
    end    

    chunked_worksize = length(chunked_arrays[1])
    
    M = worksize ÷ chunked_worksize
    loop! = _kernel!(arch, workgroup, chunked_worksize)

    for m in ProgressBar(1:M)
        idx = m * chunked_worksize - 1
        
        # Updating the chunk (copying array data to architecture(chunk)!
        for chunked_array in chunked_arrays
            update_chunk!(chunked_array, idx)
        end

        loop!(getchunk.(args)...)

        # copying back to the CPU!
        for chunked_array in chunked_arrays
            update_array!(chunked_array)
        end
    end

    return nothing
end

end