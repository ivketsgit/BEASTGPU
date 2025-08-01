# using CUDA
# using BenchmarkTools
# using KernelAbstractions


# @kernel function make_complex(complex_array, @Const(float_array))
#     i, j = @index(Global, NTuple)
#     complex_array[i, j] = Complex(float_array[1, i, j], float_array[2, i, j])
# end

# A = KernelAbstractions.ones(backend, Float64, 2, 3, 3)
# B = KernelAbstractions.zeros(CPU(), ComplexF64, 2, 3, 3)

# make_complex(backend)(B, A,ndrange=(3,3))
# KernelAbstractions.synchronize(backend)

# @show B

t = @elapsed Array{Float64}(undef, 38400, 3, 3)
@show t