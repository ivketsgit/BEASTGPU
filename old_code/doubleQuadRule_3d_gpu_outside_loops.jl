module DoubleQuadRule_3d_gpu_outside_loops
export doubleQuadRule_3d_gpu_outside_loops!

using KernelAbstractions, CUDA, Atomix
using CUDA: CuArray, CUDAKernels
using BEAST
using KernelAbstractions.Extras: @unroll


const i4pi = 1 / (4pi)
const epsilon =  eps(Float64)
# @show epsilon
@kernel function combined_kernel_temp_outside_loops!(result,
    # @Const(γ), @Const(α), 
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))

    K, L, I, J = @index(Global, NTuple)
    # GroupID = @index(Group, Linear)
    # if I == 1 && J == 1 && K == 1 && L == 1
    #     @print("\n @groupsize() = ", @groupsize() , " @ndrange() = ",@ndrange())
    # end 
    # if GroupID == 1
    #     @print("\n K, L, I, J = ",K, " ", L, " ", I," ",  J)
    # end 
    @inbounds r_1_sqr = (womps_cart[K, L, I, 1] - wimps_cart[K, L, J, 1])^2
    @inbounds r_2_sqr = (womps_cart[K, L, I, 2] - wimps_cart[K, L, J, 2])^2
    @inbounds r_3_sqr = (womps_cart[K, L, I, 3] - wimps_cart[K, L, J, 3])^2

    R = sqrt(r_1_sqr + r_2_sqr + r_3_sqr)
    @inbounds j_αG = (R == 0 ? 0 : womps_weights[K, L, I] * wimps_weights[K, L, J] * i4pi *  exp(-R*im) / R)
    
    @unroll for x in 0:2
        @inbounds j_αG_womps_values = j_αG * womps_values[K, L, I, x + 1]
        @unroll for y in 1:3
            @inbounds result[K, L, I, J, 3 * x + y] = j_αG_womps_values * wimps_values[K, L, J, y]
        end
    end
    # @inbounds j_αG = womps_weights[K, L, I] * wimps_weights[K, L, J] / (4pi * R) *  exp(-R*im) # * α * exp(-γ*R)

    # @unroll for x in 1:9
    #     quotient, remainder = divrem(x-1, 3)
    #     result[K, L, I, J, x] = j_αG * womps_values[K, L, I, quotient + 1] * wimps_values[K, L, J,  remainder + 1]
    # end


    # j_αG_womps_values_1 = j_αG * womps_values[K, L, 1, I]
    # result[K, L, 1, 1, I, J] = j_αG_womps_values_1 * wimps_values[K, L, 1, J]
    # result[K, L, 1, 2, I, J] = j_αG_womps_values_1 * wimps_values[K, L, 2, J]
    # result[K, L, 1, 3, I, J] = j_αG_womps_values_1 * wimps_values[K, L, 3, J]

    # j_αG_womps_values_2 = j_αG * womps_values[K, L, 2, I]
    # result[K, L, 2, 1, I, J] = j_αG_womps_values_2 * wimps_values[K, L, 1, J]
    # result[K, L, 2, 2, I, J] = j_αG_womps_values_2 * wimps_values[K, L, 2, J]
    # result[K, L, 2, 3, I, J] = j_αG_womps_values_2 * wimps_values[K, L, 3, J]
    
    # j_αG_womps_values_3 = j_αG * womps_values[K, L, 3, I]
    # result[K, L, 3, 1, I, J] = j_αG_womps_values_3 * wimps_values[K, L, 1, J]
    # result[K, L, 3, 2, I, J] = j_αG_womps_values_3 * wimps_values[K, L, 2, J]
    # result[K, L, 3, 3, I, J] = j_αG_womps_values_3 * wimps_values[K, L, 3, J]
end

@kernel function combined_kernel_temp_outside_loops_linear_index!(result,
    # @Const(γ), @Const(α), 
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))

    GlobalGroupId = @index(Group, Linear)
    GlobalId = @index(Global, Linear)
    # quotient_1, remainder_1 = divrem(GlobalId-1, 84 * 12)
    # quotient_2, remainder_2 = divrem(remainder_1, 12)
    # quotient_3, remainder_3 = divrem(remainder_2, 4)

    # GroupID = @index(Group, Linear)

    # K = quotient_1 + 1
    # L = quotient_2 + 1
    # I = quotient_3 + 1 
    # J = remainder_3 + 1

    
    K, L, I, J = @index(Global, NTuple)
    index_I = 1+(I−1)+(L−1)*3+(K−1)*3*84+(1−1)*84*84*3
    index_J = 1+(J−1)+(L-1)*4+(K−1)*4*84+(1−1)*84*84*4
    if GlobalGroupId == 1
        @print("\n K, L, I, J = ",K, " ", L, " ", I," ",  J, " linear = ",GlobalId, " ", " index_I = ", index_I, " index_J = ", index_J)
        # @print("\n oud womps_cart I = ", womps_cart[I, L, K, 1], " new womps_cart I = ",womps_cart[index_I])
    end 
    # if GlobalId == 1
    #     @print("\n @groupsize() = ", @groupsize() , " @ndrange() = ",@ndrange())
    # end 
    r_1 = (womps_cart[I, L, 1, K] - wimps_cart[J, L, 1, K])^2
    r_2 = (womps_cart[I, L, 2, K] - wimps_cart[J, L, 2, K])^2
    r_3 = (womps_cart[I, L, 3, K] - wimps_cart[J, L, 3, K])^2

    R = sqrt(r_1 + r_2 + r_3)
    @inbounds j_αG = (R != 0 ? womps_weights[I, L, K] * wimps_weights[J, L, K] * i4pi *  exp(-R*im) / R : 0)
    
    @unroll for x in 0:2
        @inbounds j_αG_womps_values = j_αG * womps_values[I, L, x + 1, K]
        @unroll for y in 1:3
            @inbounds result[I, J, L, 3 * x + y, K] = j_αG_womps_values * wimps_values[J, L, y, K]
        end
    end
end

@kernel function combined_kernel_temp_outside_loops_linear_index_2!(result,
    # @Const(γ), @Const(α), 
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))

    GlobalGroupId = @index(Group, Linear)
    GlobalId = @index(Global, Linear)
    quotient_1, remainder_1 = divrem(GlobalId-1, 84 * 12)
    quotient_2, remainder_2 = divrem(remainder_1, 12)
    quotient_3, remainder_3 = divrem(remainder_2, 4)

    GroupID = @index(Group, Linear)


    K = quotient_1 + 1
    L = quotient_2 + 1
    I = quotient_3 + 1 
    J = remainder_3 + 1
    if GlobalGroupId == 1
        @print("\n K, L, I, J = ",K, " ", L, " ", I," ",  J, " linear = ",GlobalId, " ")
    end 
    if GlobalId == 1
        @print("\n @groupsize() = ", @groupsize() , " @ndrange() = ",@ndrange())
    end 
    # i+(j−1)⋅84+(k−1)⋅(84⋅84)+(l−1)⋅(84⋅84⋅3)
    # womps_cart[K, L, I, 1]
    r_1 = (womps_cart[GlobalId % (84*84) + (I - 1) * 84*84            ] - wimps_cart[GlobalId % (84*84) + (J - 1) * 84*84               ])^2
    r_2 = (womps_cart[GlobalId % (84*84) + (I - 1) * 84*84 + 84*84*3  ] - wimps_cart[GlobalId % (84*84) + (J - 1) * 84*84 + 84*84*4     ])^2
    r_3 = (womps_cart[GlobalId % (84*84) + (I - 1) * 84*84 + 84*84*3*2] - wimps_cart[GlobalId % (84*84) + (J - 1) * 84*84 + 84*84*4*2   ])^2

    R = sqrt(r_1 + r_2 + r_3)
    @inbounds j_αG = (R != 0 ? womps_weights[GlobalId % (84*84) + (I - 1) * 84*84] * wimps_weights[GlobalId % (84*84) + (J - 1) * 84*84] * i4pi *  exp(-R*im) / R : 0)
    
    @unroll for x in 0:2
        @inbounds j_αG_womps_values = j_αG * womps_values[GlobalId % (84*84) + (I - 1) * 84*84 +  x * 84*84*3]
        @unroll for y in 1:3
            @inbounds result[K, L, I, J, 3 * x + y] = j_αG_womps_values * wimps_values[GlobalId % (84*84) + (J - 1) * 84*84 + 84*84*4 * (y-1)]
        end
    end
end

@kernel function combined_kernel_temp_outside_loops_linear_index_3!(result,
    # @Const(γ), @Const(α), 
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))
    
    @inbounds begin
    womps_data_local = @private Float64 (7, 3) # (1:3,3) -> cart, (4,3) -> weights, (5:7,3) -> values
    wimps_data_local = @private Float64 (7, 4) # (1:3,4) -> cart, (4,4) -> weights, (5:7,4) -> values

    K, L = @index(Global, NTuple)
    
    @unroll for I in 1:3
        womps_data_local[1,I] = womps_cart[K, L, I, 1]
        womps_data_local[2,I] = womps_cart[K, L, I, 2]
        womps_data_local[3,I] = womps_cart[K, L, I, 3]
        womps_data_local[4,I] = womps_weights[K, L, I]
        womps_data_local[5,I] = womps_values[K, L, I, 1]
        womps_data_local[6,I] = womps_values[K, L, I, 2]
        womps_data_local[7,I] = womps_values[K, L, I, 3]
    end
    @unroll for J in 1:4
        wimps_data_local[1,J] = wimps_cart[K, L, J, 1]
        wimps_data_local[2,J] = wimps_cart[K, L, J, 2]
        wimps_data_local[3,J] = wimps_cart[K, L, J, 3]
        wimps_data_local[4,J] = wimps_weights[K, L, J]
        wimps_data_local[5,J] = wimps_values[K, L, J, 1]
        wimps_data_local[6,J] = wimps_values[K, L, J, 2]
        wimps_data_local[7,J] = wimps_values[K, L, J, 3] 
    end

    result_local = @private ComplexF64 (3 * 3)
    @unroll for I in 1:9
        result_local[I] = 0
    end
    
    @unroll for I in 1:3
        @unroll for J in 1:4
            R = sqrt( (womps_data_local[1, I] - wimps_data_local[1, J])^2 + 
                                (womps_data_local[2, I] - wimps_data_local[2, J])^2 + 
                                (womps_data_local[3, I] - wimps_data_local[3, J])^2 )
            j_αG = womps_data_local[4, I] * wimps_data_local[4, J] * i4pi / R * (R != 0) * exp(-R*im) # max(R, 1e-10) to skip 0s

            @unroll for x in 0:2
                j_αG_womps_values = j_αG * womps_data_local[x + 5, I]
                @unroll for y in 1:3
                    result_local[3 * x + y] += j_αG_womps_values * wimps_data_local[y + 4,J]
                end
            end
        end
    end
    @unroll for x in 1:9
        result[K, L, x] = result_local[x]
    end
    end
end

# (84*84 + 2 ) % (84 * 84)
function doubleQuadRule_3d_gpu_outside_loops!(z, biop, size,
    womps_weights, wimps_weights, 
    womps_values, wimps_values, 
    womps_cart, wimps_cart)
    
    α = biop.alpha
    γ = biop.gamma
    if α != 1 || γ != 0.0 + 1.0im
        throw("α of γ not correct")
    end

    backend = KernelAbstractions.get_backend(z)
    time = @elapsed begin
    kernel! = combined_kernel_temp_outside_loops_linear_index_3!(backend)#, 1008, (84,3,4,84,))
    kernel!(z,
            # γ, α, 
            womps_weights, wimps_weights, 
            womps_values, wimps_values,
            womps_cart, wimps_cart, 
            ndrange = (size,size)) # , workgroupsize=1008
    KernelAbstractions.synchronize(backend)
    end
    @show time
    # time = @elapsed begin
    # if backend == CUDABackend()
    #     r = CUDA.@sync sum(z, dims=(1,2))
    # else
    #     throw("implement backend")
    # end 
    # end
    # @show time
    return z
    end
    
end