using CUDA
@inline function pinned_arr(arr, backend::CUDABackend)
    return CUDA.pin(arr)
end

@inline function pinned_arr(arr, backend::Any)
    @warn "Pinned arrays are only supported on CUDA backend. Using paged memory instead."
    return arr
end