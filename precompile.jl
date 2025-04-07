# precompile.jl
include("quadrule_determine_type.jl")

# using MyPackage  # replace with your actual package/module
# using CUDA
# using KernelAbstractions
# using KernelAbstractions.Extras: @unroll
# using GPUArrays
# using KernelAbstractions, Atomix
# using BEAST, CompScienceMeshes
# using KernelAbstractions.Extras: @unroll
# using KernelAbstractions: @atomic
# using StaticArrays


using KernelAbstractions
using CUDA

# const dtol = 2.220446049250313e-16 * 1.0e3
# const xtol2 = 0.2 * 0.2
# @kernel function quadrule_determine_type(result::AbstractArray, should_calc_gpu_::AbstractArray, @Const(k2::Float64), @Const(τ::AbstractArray), @Const(σ::AbstractArray), @Const(σ_volume::AbstractArray), @Const(type::DataType))
#     i, j = @index(Global, NTuple)

#     hits = 0
#     dmin2 = floatmax(type)
#     @unroll for unroll in 1:3
#         @unroll for unroll_ in 1:3
#             d2 = (τ[i, 1, unroll] - σ[j, 1, unroll_])^2 + 
#                  (τ[i, 2, unroll] - σ[j, 2, unroll_])^2 + 
#                  (τ[i, 3, unroll] - σ[j, 3, unroll_])^2
#             d = sqrt(d2)
#             hits += (d < dtol)
#             dmin2 = min(dmin2, d2)
#         end
#     end
    
#     h2 = σ_volume[j]
#     result[i,j] = (hits == 0 ? (max(dmin2*k2, dmin2/(16 * h2)) >= xtol2 ? 0 : 4) : hits)
    


#     # index = ((i - 1) ÷ 32 ) + 1
#     # bitindex = ((i - 1) % 32 ) + 1

#     # mask = (hits == 0) * (1 << bitindex)
#     # @print("\n UInt32(mask) = ", UInt32(mask), " ",should_calc_gpu_[index, j])
#     # @print("\n UInt32(mask) = ", mask, " ",should_calc_gpu_[index, j])
#     # b = should_calc_gpu_[index, j] | mask
#     # @print("\n b = ", b)
#     should_calc_gpu_[i, j] = hits == 0
#     # should_calc_gpu_[index, j] |= mask
# end


function precompile_entrypoint()
    
    include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend
    # Dummy inputs for compilation
    
    dummy_A = KernelAbstractions.zeros(backend, Int8, 1, 1)
    dummy_B = KernelAbstractions.zeros(backend, Int8, 1, 1)
    dummy_C = KernelAbstractions.zeros(backend, Float64, 1, 3, 3)
    dummy_D = KernelAbstractions.zeros(backend, Float64, 1, 3, 3)
    dummy_E = KernelAbstractions.zeros(backend, Float64, 1, 3)
    dummy_range = (1,1)
    # Kernel launch to force compilation

    quadrule_determine_type(backend)(dummy_A, dummy_B, 1.0, dummy_C, dummy_D, dummy_E, Float64, ndrange = dummy_range)
end

precompile_entrypoint()