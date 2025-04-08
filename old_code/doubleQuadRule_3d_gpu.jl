module DoubleQuadRule_3d_gpu
export doubleQuadRule_3d_gpu!

using KernelAbstractions, CUDA, Atomix
using CUDA: CuArray, CUDAKernels
using BEAST


const i4pi = 1 / (4pi)
@kernel function combined_kernel_temp!(result,
    @Const(γ), @Const(α), @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_cart_1), @Const(womps_cart_2), @Const(womps_cart_3), 
    @Const(wimps_cart_1), @Const(wimps_cart_2), @Const(wimps_cart_3),
    @Const(womps_values_1), @Const(womps_values_2), @Const(womps_values_3), 
    @Const(wimps_values_1), @Const(wimps_values_2), @Const(wimps_values_3))

    I, J = @index(Global, NTuple)

    j = womps_weights[I] * wimps_weights[J]
    r_1 = womps_cart_1[I] - wimps_cart_1[J]
    r_2 = womps_cart_2[I] - wimps_cart_2[J]
    r_3 = womps_cart_3[I] - wimps_cart_3[J]
    
    R = sqrt(r_1^2 + r_2^2 + r_3^2)
    j_αG = j * α[1] * exp(-γ[1]*R)*(i4pi / R)

    
    # if I == 1 && J == 1
    #     @print(
    #         "\n womps_values_1[I] = ", womps_values_1[I], " womps_values_2[I] = ", womps_values_2[I], "womps_values_3[I] = ", womps_values_3[I], 
    #         "\n wimps_values_1[I] = ", wimps_values_1[I], " wimps_values_2[I] = ", wimps_values_2[I], "wimps_values_3[I] = ", wimps_values_3[I],
    #     )
    # end
    # throw("dqsfkml")
    j_αG_womps_values_1 = j_αG * womps_values_1[I]
    result[1,I,J] = j_αG_womps_values_1 * wimps_values_1[J]
    result[2,I,J] = j_αG_womps_values_1 * wimps_values_2[J]
    result[3,I,J] = j_αG_womps_values_1 * wimps_values_3[J]

    j_αG_womps_values_2 = j_αG * womps_values_2[I]
    result[4,I,J] = j_αG_womps_values_2 * wimps_values_1[J]
    result[5,I,J] = j_αG_womps_values_2 * wimps_values_2[J]
    result[6,I,J] = j_αG_womps_values_2 * wimps_values_3[J]
    
    j_αG_womps_values_3 = j_αG * womps_values_3[I]
    result[7,I,J] = j_αG_womps_values_3 * wimps_values_1[J]
    result[8,I,J] = j_αG_womps_values_3 * wimps_values_2[J]
    result[9,I,J] = j_αG_womps_values_3 * wimps_values_3[J]

    # if I == 1 && J == 1
    #     x = 0
    #     for i in 1:9
    #         x += result[i,I,J]
    #     end
        
    #     @print("\n x = ", real(x), " " ,imag(x))
    #     throw("kqmsf")
    # end
end  



function doubleQuadRule_3d_gpu!(biop, tshs, bshs, tcell, bcell, z, strat)
    # println("momintegrals_gpu_doubleQuad_3d")
    γ = CuArray{ComplexF64}(hcat(biop.gamma...))
    α = CuArray{ComplexF64}(hcat(biop.alpha...))
    # womps_values = CuArray{ComplexF64}(hcat(map(x -> collect(x), biop.alpha)...))

    womps = strat.outer_quad_points
    wimps = strat.inner_quad_points

    include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

    data = womps
    #WEIGHT
    womps_weights = CuArray(([entry.weight for entry in data]))

    #VALUES
    womps_values_1 =  CuArray(([entry.value[1].value for entry in data]))
    womps_values_2 =  CuArray(([entry.value[2].value for entry in data]))
    womps_values_3 =  CuArray(([entry.value[3].value for entry in data]))


    # @show womps_values_1
    # @show womps_values_2
    # @show womps_values_3
    # throw("qsdf")

    #POINT
    womps_cart_1 =   CuArray(([entry.point.cart[1] for entry in data]))
    womps_cart_2 =   CuArray(([entry.point.cart[2] for entry in data]))
    womps_cart_3 =   CuArray(([entry.point.cart[3] for entry in data]))

    womps_cart = CuArray{Float64}(hcat([entry.point.cart for entry in data]...))

    # @show womps_cart_1
    # @show womps_cart_2
    # @show womps_cart_3
    # @show womps_cart
    # throw("dkqsfmjqsdlkf")

    data = wimps
    #WEIGHT
    wimps_weights = CuArray(([entry.weight for entry in data]))

    #VALUES
    wimps_values_1 =  CuArray(([entry.value[1].value for entry in data]))
    wimps_values_2 =  CuArray(([entry.value[2].value for entry in data]))
    wimps_values_3 =  CuArray(([entry.value[3].value for entry in data]))

    #POINT
    wimps_cart_1 =   CuArray(([entry.point.cart[1] for entry in data]))
    wimps_cart_2 =   CuArray(([entry.point.cart[2] for entry in data]))
    wimps_cart_3 =   CuArray(([entry.point.cart[3] for entry in data]))

    len_womps = length(womps_cart_1)
    len_wimps = length(wimps_cart_1)
    
    
    result = KernelAbstractions.zeros(backend, ComplexF64, 9, len_womps, len_wimps)

    kernel! = combined_kernel_temp!(backend)
    kernel!(result,
            γ, α, womps_weights, wimps_weights, 
            womps_cart_1, womps_cart_2, womps_cart_3,
            wimps_cart_1, wimps_cart_2, wimps_cart_3,
            womps_values_1, womps_values_2, womps_values_3,
            wimps_values_1, wimps_values_2, wimps_values_3,
            ndrange = (len_womps,len_wimps))
    KernelAbstractions.synchronize(backend)
    
    if backend == CUDABackend()
        r = CUDA.@sync sum(result, dims=(2,3))
        cpu_array = Array(r)
        # @show size(result)
        # @show Array(result)[1,:,:]
        # @show sum(Array(result)[1,:,:])
        # @show r
        # throw("jkqfml")
        cpu_array = permutedims(reshape(cpu_array, 3, 3), (2, 1))
    else
        throw("implement backend")
    end 

    M = length(womps[1].value)
    N = length(wimps[1].value)
    z[1:M, 1:N] .= cpu_array
    # throw("kdslqmfqjds")
    return cpu_array
    end
    
end




