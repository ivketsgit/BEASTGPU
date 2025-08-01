using KernelAbstractions
using CUDA
using BenchmarkTools
include("../utils/backend.jl")

size_in_GIB = 2 * 38402* 38402 * sizeof(Float64) / 2^30
arr_cuda = Array{Float64}(undef, 2, 38402, 38402)
pinned_arr = CUDA.pin(arr_cuda)
gpu_array = CuArray{Float64}(undef, 2, 38402, 38402)


@kernel function dumy(arr)
    i, j, k = @index(Global, NTuple)
    arr[i, j, k] = 1.0
end

dumy(backend)(gpu_array, ndrange = (2, 38402, 38402))
KernelAbstractions.synchronize(backend)

time = @belapsed begin copyto!(pinned_arr, gpu_array) end
println("time: ", time)
println("GIB/S: ", size_in_GIB/time)

@btime copyto!(pinned_arr, gpu_array)
@show sum(pinned_arr) == prod((2, 38402, 38402))

@btime Array{Float64}(undef, 2, 38402, 38402)
paged_arr = KernelAbstractions.allocate(backend, Float64, 2, 38402, 38402)
arr_kern = Array{Float64}(undef, 2, 38402, 38402)
dumy(backend)(paged_arr, ndrange = (2, 38402, 38402))
KernelAbstractions.synchronize(backend)

# @btime Array{Float64}(undef, size(paged_arr))
arr_kern = Array{Float64}(undef, size(paged_arr))

# @btime copyto!(arr_kern, paged_arr)
time = @belapsed begin copyto!(arr_kern, paged_arr) end
println("time: ", time)
println("GIB/S: ", size_in_GIB/time)



time = @belapsed begin copyto!(arr_kern, paged_arr) end
println("time: ", time)
println("GIB/S: ", size_in_GIB/time)


time = @belapsed begin Array(paged_arr) end
println("time: ", time)
println("GIB/S: ", size_in_GIB/time)


# @btime Array(paged_arr)
@show sum(arr_kern) == prod((2, 38402, 38402))