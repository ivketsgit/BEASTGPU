
const i4pi = 1 / (4pi)
const epsilon =  eps(Float64)


@kernel function combined_kernel_temp_outside_loops_linear_index_and_store!(result,
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(size),
    @Const(γ), @Const(α),
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))
    
    @inbounds begin
        womps_data_local = @private Float64 (7, 3) # (1:3,3) -> cart, (4,3) -> weights, (5:7,3) -> values
        wimps_data_local = @private Float64 (7, 4) # (1:3,4) -> cart, (4,4) -> weights, (5:7,4) -> values

        K, L = @index(Global, NTuple)
        
        @unroll for I in 1:3
            womps_data_local[1,I] = womps_cart[K, I, 1]
            womps_data_local[2,I] = womps_cart[K, I, 2]
            womps_data_local[3,I] = womps_cart[K, I, 3]
            womps_data_local[4,I] = womps_weights[K, I]
            womps_data_local[5,I] = womps_values[K, I, 1]
            womps_data_local[6,I] = womps_values[K, I, 2]
            womps_data_local[7,I] = womps_values[K, I, 3]
        end
        @unroll for J in 1:4
            wimps_data_local[1,J] = wimps_cart[L, J, 1]
            wimps_data_local[2,J] = wimps_cart[L, J, 2]
            wimps_data_local[3,J] = wimps_cart[L, J, 3]
            wimps_data_local[4,J] = wimps_weights[L, J]
            wimps_data_local[5,J] = wimps_values[L, J, 1]
            wimps_data_local[6,J] = wimps_values[L, J, 2]
            wimps_data_local[7,J] = wimps_values[L, J, 3] 
        end

        result_local = @private ComplexF64 (3 * 3)
        @unroll for I in 1:9
            result_local[I] = 0
        end
        
        @unroll for I in 1:3
            @unroll for J in 1:4
                @fastmath R = sqrt( (womps_data_local[1, I] - wimps_data_local[1, J])^2 + 
                                    (womps_data_local[2, I] - wimps_data_local[2, J])^2 + 
                                    (womps_data_local[3, I] - wimps_data_local[3, J])^2 )
                j_αG = α * womps_data_local[4, I] * wimps_data_local[4, J] * i4pi / max(R, 1e-10) * exp(-R*γ) # max(R, 1e-10) to skip 0s should_calc[K,L] * 

                @unroll for x in 0:2
                    j_αG_womps_values = j_αG * womps_data_local[x + 5, I]
                    @unroll for y in 1:3
                        result_local[3 * x + y] += j_αG_womps_values * wimps_data_local[y + 4,J]
                    end
                end
            end
        end
        store_with_kernel!(result, test_assembly_gpu, trial_assembly_gpu, result_local, K, L)
    end
end

@kernel function combined_kernel_temp_outside_loops_linear_index_and_store_partial_reduced_mem!(result,
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(size),
    @Const(γ), @Const(α),
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))
    
    @inbounds begin
        K, L = @index(Global, NTuple)

        result_local = @private ComplexF64 (9)
        @unroll for I in 1:9
            result_local[I] = 0
        end
        
        @unroll for I in 1:3
            @unroll for J in 1:4
                @fastmath R = sqrt( (womps_cart[K, I, 1] - wimps_cart[L, J, 1])^2 + 
                                    (womps_cart[K, I, 2] - wimps_cart[L, J, 2])^2 + 
                                    (womps_cart[K, I, 3] - wimps_cart[L, J, 3])^2 )
                j_αG = α * womps_weights[K, I] * wimps_weights[L, J] * i4pi / max(R, 1e-10) * exp(-R*γ) # max(R, 1e-10) to skip 0s should_calc[K,L] * 

                @unroll for x in 0:2
                    j_αG_womps_values = j_αG * womps_values[K, I, x + 1]
                    @unroll for y in 1:3
                        result_local[3 * x + y] += j_αG_womps_values * wimps_values[L, J, y]
                    end
                end
            end
        end
        store_with_kernel!(result, test_assembly_gpu, trial_assembly_gpu, result_local, K, L)
    end
end

@kernel function combined_kernel_temp_outside_loops_linear_index_and_store_reduced_mem!(result,
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(size),
    @Const(γ), @Const(α),
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))
    
    K, L = @index(Global, NTuple)

    @inbounds begin
        R1 = 0
        R2 = 0
        R3 = 0
        R4 = 0
        R5 = 0
        R6 = 0
        R7 = 0
        R8 = 0
        R9 = 0
        
        @unroll for I in 1:3
            @unroll for J in 1:4
                @fastmath R = sqrt( (womps_cart[K, I, 1] - wimps_cart[L, J, 1])^2 + 
                                    (womps_cart[K, I, 2] - wimps_cart[L, J, 2])^2 + 
                                    (womps_cart[K, I, 3] - wimps_cart[L, J, 3])^2 )
                j_αG = α * womps_weights[K, I] * wimps_weights[L, J] * i4pi / max(R, 1e-10) * exp(-R*γ) # max(R, 1e-10) to skip 0s should_calc[K,L] * 

                j_αG_womps_values = j_αG * womps_values[K, I, 1]
                R1 += j_αG_womps_values * wimps_values[L, J, 1]
                R2 += j_αG_womps_values * wimps_values[L, J, 2]
                R3 += j_αG_womps_values * wimps_values[L, J, 3]

                j_αG_womps_values = j_αG * womps_values[K, I, 2]
                R4 += j_αG_womps_values * wimps_values[L, J, 1]
                R5 += j_αG_womps_values * wimps_values[L, J, 2]
                R6 += j_αG_womps_values * wimps_values[L, J, 3]

                j_αG_womps_values = j_αG * womps_values[K, I, 3]
                R7 += j_αG_womps_values * wimps_values[L, J, 1]
                R8 += j_αG_womps_values * wimps_values[L, J, 2]
                R9 += j_αG_womps_values * wimps_values[L, J, 3]
            end
        end
        store_with_kernel_register!(result, test_assembly_gpu, trial_assembly_gpu, K, L, R1,R2,R3,R4,R5,R6,R7,R8,R9)
    end
end

function calc_index_block(number, size)
    x, y = divrem(number - 1, 96)
    Block_y, local_y  = divrem(y, 32)
    Block_x, local_x  = divrem(x, 32)
    should_calculate = (x + 1 <= size && y + 1 <= size) ? true : false

    return Block_y + 1, local_y + 1, Block_x + 1, local_x + 1, y + 1, x + 1, should_calculate
end


@kernel function combined_kernel_temp_outside_loops_block_and_store!(result, 
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(size),
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))
    
    # @inbounds begin
    womps_data_local = @localmem Float64 (32, 7, 3) # (1:3,3) -> cart, (4,3) -> weights, (5:7,3) -> values
    wimps_data_local = @localmem Float64 (32, 7, 4) # (1:3,4) -> cart, (4,4) -> weights, (5:7,4) -> values

    Group_number = @index(Group, Linear)
    Global_number = @index(Global, Linear)

    Block_y, local_y, Block_x, local_x, K, L, should_calculate = calc_index_block(Global_number, size)

    K = should_calculate ? K : 84
    L = should_calculate ? L : 84

    # if Block_y == 1 && Block_x == 2 && local_y == 1
    #     @print("\n local_x = ", local_x, "    K = ", K, " L = ", L)
    # end
    
    # if Group_number == 1
    #     @print("\n Global_number = ", Global_number, "      Block_y = ", Block_y, " Block_x = ", Block_x,"     local_y = ", local_y, " local_x = ", local_x, "    K = ", K, " L = ", L)
    # end

    
    # @inbounds begin
        if local_x == 1
            if Block_y == 1 && Block_x == 1
                @print("\n local_y = ", local_y, " K = ", K)
            end
            @unroll for I in 1:3
                womps_data_local[local_y, 1,I] = womps_cart[K, I, 1]
                womps_data_local[local_y, 2,I] = womps_cart[K, I, 2]
                womps_data_local[local_y, 3,I] = womps_cart[K, I, 3]
                womps_data_local[local_y, 4,I] = womps_weights[K, I]
                womps_data_local[local_y, 5,I] = womps_values[K, I, 1]
                womps_data_local[local_y, 6,I] = womps_values[K, I, 2]
                womps_data_local[local_y, 7,I] = womps_values[K, I, 3]
            end
        end
        if local_y == 1
            @unroll for J in 1:4
                wimps_data_local[local_x, 1,J] = wimps_cart[L, J, 1]
                wimps_data_local[local_x, 2,J] = wimps_cart[L, J, 2]
                wimps_data_local[local_x, 3,J] = wimps_cart[L, J, 3]
                wimps_data_local[local_x, 4,J] = wimps_weights[L, J]
                wimps_data_local[local_x, 5,J] = wimps_values[L, J, 1]
                wimps_data_local[local_x, 6,J] = wimps_values[L, J, 2]
                wimps_data_local[local_x, 7,J] = wimps_values[L, J, 3] 
            end
        end
        @synchronize

        result_local = @private ComplexF64 (3 * 3)
        @unroll for unroll in 1:9
            result_local[unroll] = 0
        end
        
        if K == 1 && L == 1
            @unroll for I in 1:3
                @unroll for J in 1:4
                    @fastmath R = sqrt( (womps_data_local[local_y, 1, I] - wimps_data_local[local_x, 1, J])^2 + 
                                        (womps_data_local[local_y, 2, I] - wimps_data_local[local_x, 2, J])^2 + 
                                        (womps_data_local[local_y, 3, I] - wimps_data_local[local_x, 3, J])^2 )
                    j_αG = should_calculate * womps_data_local[local_y, 4, I] * wimps_data_local[local_x, 4, J] * i4pi / max(R, 1e-10) * exp(-R*im) # max(R, 1e-10) to skip 0s should_calc[K,L] *

                    @unroll for x in 0:2
                        j_αG_womps_values = j_αG * womps_data_local[local_y, x + 5, I]
                        @unroll for y in 1:3
                            result_local[3 * x + y] += j_αG_womps_values * wimps_data_local[local_x, y + 4,J]
                        end
                    end
                end
            end
            @print("\n local_y = ", local_y, " K = ", K, " local_x = ", local_x, " L = ", L)
            @print("\n womps_cart[K, I, 1] = ", womps_cart[K, 1, 1]," womps_cart[K, 2, 1] = ", womps_cart[K, 2, 1]," womps_cart[K, 3, 1] = ", womps_cart[K, 3, 1])
            @print("\n womps_cart[K, I, 2] = ", womps_cart[K, 1, 2]," womps_cart[K, 2, 2] = ", womps_cart[K, 2, 2]," womps_cart[K, 3, 2] = ", womps_cart[K, 3, 2])
            @print("\n womps_cart[K, I, 3] = ", womps_cart[K, 1, 3]," womps_cart[K, 2, 3] = ", womps_cart[K, 2, 3]," womps_cart[K, 3, 3] = ", womps_cart[K, 3, 3])
            @unroll for I in 1:3
                @print("\n womps_data_local = ", womps_data_local[local_y, 1, I], " ", womps_data_local[local_y, 2, I], " ", womps_data_local[local_y, 3, I])
            end
            @unroll for J in 1:4
                @print("\n wimps_data_local = ", wimps_data_local[local_x, 1, J], " ", wimps_data_local[local_x, 2, J], " ", wimps_data_local[local_x, 3, J])
            end
            for unroll in 1:9
                @print("\n result_local = ", real(result_local[unroll]) * 10 ^5, " ", imag(result_local[unroll]) * 10 ^5)
            end
        end
        store_with_kernel!(result, test_assembly_gpu, trial_assembly_gpu, result_local, K, L)
    # end
end

function doubleQuadRule_3d_gpu_outside_loops_and_store!(result, 
    test_assembly_gpu, trial_assembly_gpu,
    size,
    biop,
    womps_weights, wimps_weights, 
    womps_values, wimps_values, 
    womps_cart, wimps_cart)

    α = biop.alpha
    γ = biop.gamma
    
    time = @elapsed begin
        backend = KernelAbstractions.get_backend(result)
        kernel! = combined_kernel_temp_outside_loops_linear_index_and_store_reduced_mem!(backend)#, 1008, (84,3,4,84,))
        kernel!(result,
                test_assembly_gpu, trial_assembly_gpu,
                size,
                γ, α,
                womps_weights, wimps_weights, 
                womps_values, wimps_values,
                womps_cart, wimps_cart, 
                ndrange = (size, size)) # , workgroupsize=1008
        KernelAbstractions.synchronize(backend)
    end
    return time
    # @show time
    # time = @elapsed begin
    # if backend == CUDABackend()
    #     r = CUDA.@sync sum(z, dims=(1,2))
    # else
    #     throw("implement backend")
    # end 
    # end
    # @show time
    # return z
end

@kernel function combined_kernel_temp_outside_loops_linear_index_and_store_repaire!(result, 
    @Const(store_index),
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(γ), @Const(α),
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))

    # @inbounds begin
        womps_data_local = @private Float64 (7, 3) # (1:3,3) -> cart, (4,3) -> weights, (5:7,3) -> values
        wimps_data_local = @private Float64 (7, 4) # (1:3,4) -> cart, (4,4) -> weights, (5:7,4) -> values
    
        index = @index(Global, Linear)[1]
        K = store_index[index, 1]
        L = store_index[index, 2]
        
        @unroll for I in 1:3
            womps_data_local[1,I] = womps_cart[K, I, 1]
            womps_data_local[2,I] = womps_cart[K, I, 2]
            womps_data_local[3,I] = womps_cart[K, I, 3]
            womps_data_local[4,I] = womps_weights[K, I]
            womps_data_local[5,I] = womps_values[K, I, 1]
            womps_data_local[6,I] = womps_values[K, I, 2]
            womps_data_local[7,I] = womps_values[K, I, 3]
        end
        @unroll for J in 1:4
            wimps_data_local[1,J] = wimps_cart[L, J, 1]
            wimps_data_local[2,J] = wimps_cart[L, J, 2]
            wimps_data_local[3,J] = wimps_cart[L, J, 3]
            wimps_data_local[4,J] = wimps_weights[L, J]
            wimps_data_local[5,J] = wimps_values[L, J, 1]
            wimps_data_local[6,J] = wimps_values[L, J, 2]
            wimps_data_local[7,J] = wimps_values[L, J, 3] 
        end
    
        result_local = @private ComplexF64 (3 * 3)
        @unroll for I in 1:9
            result_local[I] = 0
        end
        
        @unroll for I in 1:3
            @unroll for J in 1:4
                @fastmath R = sqrt( (womps_data_local[1, I] - wimps_data_local[1, J])^2 + 
                                    (womps_data_local[2, I] - wimps_data_local[2, J])^2 + 
                                    (womps_data_local[3, I] - wimps_data_local[3, J])^2 )
                j_αG = womps_data_local[4, I] * wimps_data_local[4, J] * i4pi / max(R, 1e-10) * exp(-R*im) # max(R, 1e-10) to skip 0s should_calc[K,L] * 
    
                @unroll for x in 0:2
                    j_αG_womps_values = -j_αG * womps_data_local[x + 5, I]
                    @unroll for y in 1:3
                        result_local[3 * x + y] += j_αG_womps_values * wimps_data_local[y + 4,J]
                    end
                end
            end
        end
        
        store_with_kernel!(result, test_assembly_gpu, trial_assembly_gpu, result_local, K, L)
    # end
end

@kernel function combined_kernel_temp_outside_loops_linear_index_and_store_repaire_reduce_mem!(result, 
    @Const(store_index),
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(γ), @Const(α),
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart))

    # @inbounds begin
    womps_data_local = @private Float64 (7, 3) # (1:3,3) -> cart, (4,3) -> weights, (5:7,3) -> values
    wimps_data_local = @private Float64 (7, 4) # (1:3,4) -> cart, (4,4) -> weights, (5:7,4) -> values

    
    @inbounds begin
        index = @index(Global, Linear)[1]
        K = store_index[index, 1]
        L = store_index[index, 2]

        R1 = 0
        R2 = 0
        R3 = 0
        R4 = 0
        R5 = 0
        R6 = 0
        R7 = 0
        R8 = 0
        R9 = 0
        
        @unroll for I in 1:3
            @unroll for J in 1:4
                @fastmath R = sqrt( (womps_cart[K, I, 1] - wimps_cart[L, J, 1])^2 + 
                                    (womps_cart[K, I, 2] - wimps_cart[L, J, 2])^2 + 
                                    (womps_cart[K, I, 3] - wimps_cart[L, J, 3])^2 )
                j_αG = - α * womps_weights[K, I] * wimps_weights[L, J] * i4pi / max(R, 1e-10) * exp(-R*γ) # max(R, 1e-10) to skip 0s should_calc[K,L] * 

                j_αG_womps_values = j_αG * womps_values[K, I, 1]
                R1 += j_αG_womps_values * wimps_values[L, J, 1]
                R2 += j_αG_womps_values * wimps_values[L, J, 2]
                R3 += j_αG_womps_values * wimps_values[L, J, 3]

                j_αG_womps_values = j_αG * womps_values[K, I, 2]
                R4 += j_αG_womps_values * wimps_values[L, J, 1]
                R5 += j_αG_womps_values * wimps_values[L, J, 2]
                R6 += j_αG_womps_values * wimps_values[L, J, 3]

                j_αG_womps_values = j_αG * womps_values[K, I, 3]
                R7 += j_αG_womps_values * wimps_values[L, J, 1]
                R8 += j_αG_womps_values * wimps_values[L, J, 2]
                R9 += j_αG_womps_values * wimps_values[L, J, 3]
            end
        end
        store_with_kernel_register!(result, test_assembly_gpu, trial_assembly_gpu, K, L, R1,R2,R3,R4,R5,R6,R7,R8,R9, T2)
    end
end

function doubleQuadRule_3d_gpu_repaire!(result, 
    SauterSchwabQuadratureCommonVertex,
    SauterSchwabQuadratureCommonEdge,
    SauterSchwabQuadratureCommonFace,
    test_assembly_gpu, trial_assembly_gpu,
    biop,
    womps_weights, wimps_weights, 
    womps_values, wimps_values, 
    womps_cart, wimps_cart)


    α = biop.alpha
    γ = biop.gamma

    # @show length_store_index_CommonVertex, length_store_index_CommonEdge, length_store_index_CommonFace
    time = @elapsed begin
        length_store_index_CommonVertex = length(SauterSchwabQuadratureCommonVertex.store_index)
        length_store_index_CommonEdge = length(SauterSchwabQuadratureCommonEdge.store_index)
        length_store_index_CommonFace = length(SauterSchwabQuadratureCommonFace.store_index)
        store_index = Array{Int64}(undef,length_store_index_CommonVertex + length_store_index_CommonEdge + length_store_index_CommonFace,2)
        for i in 1:length_store_index_CommonVertex
            for j in 1:2
                store_index[i,j] = SauterSchwabQuadratureCommonVertex.store_index[i][j]
            end
        end
        for i in 1:length_store_index_CommonEdge
            for j in 1:2
                store_index[i + length_store_index_CommonVertex,j] = SauterSchwabQuadratureCommonEdge.store_index[i][j]
            end
        end
        for i in 1:length_store_index_CommonFace
            for j in 1:2
                store_index[i + length_store_index_CommonVertex + length_store_index_CommonEdge,j] = SauterSchwabQuadratureCommonFace.store_index[i][j]
            end
        end

        backend = KernelAbstractions.get_backend(result)
        store_index = move(backend, store_index)
    end
    time_2 = @elapsed begin
        kernel! = combined_kernel_temp_outside_loops_linear_index_and_store_repaire_reduce_mem!(backend)#, 1008, (84,3,4,84,))
        kernel!(result, 
                store_index,
                test_assembly_gpu, trial_assembly_gpu,
                γ, α,
                womps_weights, wimps_weights, 
                womps_values, wimps_values,
                womps_cart, wimps_cart, 
                ndrange = (length_store_index_CommonVertex + length_store_index_CommonEdge + length_store_index_CommonFace)) # , workgroupsize=1008
        KernelAbstractions.synchronize(backend)
    end
    return time, time_2
end




function doubleQuadRule_generic_3d_gpu_outside_loop_square_base_algorithm!(result,
    test_assembly_gpu, trial_assembly_gpu,
    size_qrule,
    biop,
    womps_weights, wimps_weights, 
    womps_values, wimps_values, 
    womps_cart, wimps_cart, instance::doubleQuadRuleGpuStrategy, time_table, index, ndrange_,
    should_calc_=0,
    SauterSchwabQuadratureCommonVertex=0, SauterSchwabQuadratureCommonEdge=0,SauterSchwabQuadratureCommonFace=0)

    backend = KernelAbstractions.get_backend(result)
    store_index = load_data_(instance, backend, time_table, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
    
    α = biop.alpha
    γ = biop.gamma
    should_calc = move(backend, should_calc_)

    time_1 = @elapsed begin
        backend = KernelAbstractions.get_backend(result)
        time = @elapsed begin
        #main part of matrix
        x_offset = 0
        y_offset = 0

        kernel! = combined_kernel_temp_outside_loops_linear_index_square_base_algorithm!(backend)
        kernel!(result,
                test_assembly_gpu, trial_assembly_gpu,
                size_qrule,
                γ, α,
                womps_weights, wimps_weights, 
                womps_values, wimps_values,
                womps_cart, wimps_cart, 
                instance,
                should_calc,
                store_index,
                ndrange = floor32(ndrange_[1])*floor32(ndrange_[2]))
        KernelAbstractions.synchronize(backend)
        end
        @show time
        time = @elapsed begin

        #right part of the matrix
        x_offset = floor32(ndrange_[1])
        kernel! = combined_kernel_temp_outside_loops_linear_index!(backend)
        kernel!(result,
                test_assembly_gpu, trial_assembly_gpu,
                size_qrule,
                γ, α,
                womps_weights, wimps_weights, 
                womps_values, wimps_values,
                womps_cart, wimps_cart, 
                x_offset, 0,
                instance,
                should_calc,
                store_index,
                ndrange = (ndrange_[1]-x_offset, floor32(ndrange_[2])))
        KernelAbstractions.synchronize(backend)
        end
        @show time
        time = @elapsed begin

        #bottom part of the matrix
        y_offset = floor32(ndrange_[2])
        kernel! = combined_kernel_temp_outside_loops_linear_index!(backend)
        kernel!(result,
                test_assembly_gpu, trial_assembly_gpu,
                size_qrule,
                γ, α,
                womps_weights, wimps_weights, 
                womps_values, wimps_values,
                womps_cart, wimps_cart, 
                0, y_offset,
                instance,
                should_calc,
                store_index,
                ndrange = (ndrange_[1], ndrange_[2] - y_offset))
        KernelAbstractions.synchronize(backend)
        end
        @show time
    end

    time_table[2,index] += time_1
end

const WARP_SIZE = 32  
# const WARP_SHIFT = Int(log2(WARP_SIZE))   
@inline ceil32(n) = (n + WARP_SIZE-1) & -WARP_SIZE
@inline floor32(n) = n & -WARP_SIZE
function calc_index_block(number, size)
    y, x = divrem(number - 1, floor32(size))
    
    # Block_y = y >> WARP_SHIFT      # Equivalent to div(y, 32)
    local_y = y & (WARP_SIZE - 1)  # Equivalent to rem(y, 32)
    # Block_x = x >> WARP_SHIFT       # Equivalent to div(x, 32)
    local_x = x & (WARP_SIZE - 1)  # Equivalent to rem(x, 32)
    
    return local_y + 1, local_x + 1, y + 1, x + 1
end

# println("new test")
# for i in 1:100
#     Block_y, local_y, Block_x, local_x, y, x = calc_index_block(i, 64)
#     @show (Block_x, Block_y), (local_x, local_y), (x, y)
# end

@kernel function combined_kernel_temp_outside_loops_linear_index_square_base_algorithm!(result,
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(size),
    @Const(γ), @Const(α),
    @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_values), @Const(wimps_values),
    @Const(womps_cart), @Const(wimps_cart),
    T::doubleQuadRuleGpuStrategy, @Const(should_calc), @Const(store_index))    

    gloabal_index = @index(Global, Linear)
    # @print("\n ")
    local_y, local_x, y, x = calc_index_block(gloabal_index, size)

    # womps_cart_local = @localmem Float64 (32,3,3)
    # wimps_cart_local = @localmem Float64 (32,4,3)


    # womps_weights_local = @localmem Float64 (32,3)
    # wimps_weights_local = @localmem Float64 (32,4)

    # womps_values_local = @localmem Float64 (32,3,3)
    # wimps_values_local = @localmem Float64 (32,4,3)

    # if Block_x == 1 && Block_y == 1
    #     @print("\n local_x, local_y = ", local_x, " ", local_y)
    # end

    # @synchronize

    # if local_y == 1
    #     @unroll for I in 1:3
    #         @unroll for Unroll in 1:3
    #             womps_cart_local[local_x, I, Unroll] = womps_cart[x, I, Unroll]
    #             womps_values_local[local_x, I, Unroll] = womps_values[x, I, Unroll]
    #         end
    #         womps_weights_local[local_x, I] = womps_weights[x, I]
    #     end
    # # end
    # # if local_x == 1 
    #     @unroll for J in 1:4
    #         @unroll for Unroll in 1:3
    #             wimps_cart_local[local_y, J, Unroll] = wimps_cart[y, J, Unroll]
    #             wimps_values_local[local_y, J, Unroll] = wimps_values[y, J, Unroll]
    #         end
    #         wimps_weights_local[local_y, J] = wimps_weights[y, J]
    #     end
    # # end

    # @synchronize

    # @inbounds begin
        R1 = 0
        R2 = 0
        R3 = 0
        R4 = 0
        R5 = 0
        R6 = 0
        R7 = 0
        R8 = 0
        R9 = 0
        
        @unroll for I in 1:3
            @unroll for J in 1:4
                R = 0
                @unroll for Unroll in 1:3
                    @fastmath R += (womps_cart[x, I, Unroll] - wimps_cart[y, J, Unroll])^2
                end
                R = sqrt(R)
                j_αG = calc_j_αG(α, womps_weights[x, I], wimps_weights[y, J], R, γ, T, should_calc, x, y)

                j_αG_womps_values = j_αG * womps_values[x, I, 1]
                R1 += j_αG_womps_values * wimps_values[y, J, 1]
                R2 += j_αG_womps_values * wimps_values[y, J, 2]
                R3 += j_αG_womps_values * wimps_values[y, J, 3]

                j_αG_womps_values = j_αG * womps_values[x, I, 2]
                R4 += j_αG_womps_values * wimps_values[y, J, 1]
                R5 += j_αG_womps_values * wimps_values[y, J, 2]
                R6 += j_αG_womps_values * wimps_values[y, J, 3]

                j_αG_womps_values = j_αG * womps_values[x, I, 3]
                R7 += j_αG_womps_values * wimps_values[y, J, 1]
                R8 += j_αG_womps_values * wimps_values[y, J, 2]
                R9 += j_αG_womps_values * wimps_values[y, J, 3]
            end
        end
        store_with_kernel_register!(result, test_assembly_gpu, trial_assembly_gpu, x, y, R1,R2,R3,R4,R5,R6,R7,R8,R9)
        
    # end
end