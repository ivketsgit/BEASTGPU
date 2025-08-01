using KernelAbstractions.Extras: @unroll
using KernelAbstractions: @atomic

using KernelAbstractions
include("CustomDataStructs/GpuWriteBack.jl")

@inline function store_with_kernel!(result, test_assembly_gpu, trial_assembly_gpu, result_local, K, L)
    @unroll for k in 1:9
        quotient, remainder  = divrem(k-1, 3)
        # @print("\n K = ", K, " L = ", L)
        @atomic result[1, test_assembly_gpu[quotient + 1, K][1], trial_assembly_gpu[remainder + 1, L][1]] += real(result_local[k]) * test_assembly_gpu[quotient + 1, K][2] * trial_assembly_gpu[remainder + 1, L][2]
        @atomic result[2, test_assembly_gpu[quotient + 1, K][1], trial_assembly_gpu[remainder + 1, L][1]] += imag(result_local[k]) * test_assembly_gpu[quotient + 1, K][2] * trial_assembly_gpu[remainder + 1, L][2]
    end
end

@inline function store_with_kernel_splits!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, igd_Integrands, K, L, T::GpuWriteBackTrue, Global_number)
    @unroll for k in 0:8
        remainder, quotient = divrem(k, 3)
        # @print("\n quotient = ", quotient, " remainder = ",remainder, "   K = ",  K, " L = ", L, " igd_Integrands[k * 2 + 1] = ",igd_Integrands[k * 2 + 1] , " igd_Integrands[k * 2 + 2] = ", igd_Integrands[k * 2 + 2])
        
        @atomic result[1, test_assembly_gpu_indexes[quotient + 1, K], trial_assembly_gpu_indexes[remainder + 1, L]] += igd_Integrands[k * 2 + 1] * test_assembly_gpu_values[quotient + 1, K] * trial_assembly_gpu_values[remainder + 1, L]
        @atomic result[2, test_assembly_gpu_indexes[quotient + 1, K], trial_assembly_gpu_indexes[remainder + 1, L]] += igd_Integrands[k * 2 + 2] * test_assembly_gpu_values[quotient + 1, K] * trial_assembly_gpu_values[remainder + 1, L]
    end
end

@inline function store_with_kernel_splits__!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, igd_Integrands, K, L, T::GpuWriteBackTrue, Global_number)
    @unroll for k in 0:8
        remainder, quotient = divrem(k, 3)
        # @print("\n quotient = ", quotient, " remainder = ",remainder, "   K = ",  K, " L = ", L, " igd_Integrands[k * 2 + 1] = ",igd_Integrands[k * 2 + 1] , " igd_Integrands[k * 2 + 2] = ", igd_Integrands[k * 2 + 2])
        
        @atomic result[1, test_assembly_gpu_indexes[quotient + 1, K], trial_assembly_gpu_indexes[remainder + 1, L]] += igd_Integrands[k * 256 * 2       + 1] * test_assembly_gpu_values[quotient + 1, K] * trial_assembly_gpu_values[remainder + 1, L]
        @atomic result[2, test_assembly_gpu_indexes[quotient + 1, K], trial_assembly_gpu_indexes[remainder + 1, L]] += igd_Integrands[k * 256 * 2 + 256 + 1] * test_assembly_gpu_values[quotient + 1, K] * trial_assembly_gpu_values[remainder + 1, L]
    end
end

@inline function store_with_kernel_splits!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, igd_Integrands, K, L, T::GpuWriteBackFalse, Global_number)
    @unroll for k in 1:9
        remainder, quotient = divrem(k - 1, 3)
        result[(Global_number - 1) >> 8 + 1, k] = (igd_Integrands[(k-1) * 2 + 1] + igd_Integrands[(k-1) * 2 + 2] * im)  * test_assembly_gpu_values[quotient + 1, K] * trial_assembly_gpu_values[remainder + 1, L]
    end
end


@inline function store_with_kernel_splits__!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes,
                                             test_assembly_gpu_values, trial_assembly_gpu_values,
                                             igd_Integrands, K, L, T::GpuWriteBackFalse, Global_number)
    @unroll for k in 1:9
        remainder, quotient = divrem(k - 1, 3)
        real_part = igd_Integrands[(k - 1) * 256 * 2       + 1]
        imag_part = igd_Integrands[(k - 1) * 256 * 2 + 256 + 1]
        val = (real_part + imag_part * im) *
              test_assembly_gpu_values[quotient + 1, K] *
              trial_assembly_gpu_values[remainder + 1, L]
        result[(Global_number - 1) >> 8 + 1, k] = val
    end
end

@inline function store_with_kernel_register!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, K, L, x_offset, y_offset, R1,R2,R3,R4,R5,R6,R7,R8,R9, T::GpuWriteBackTrue)
    @atomic result[1, test_assembly_gpu_indexes[1, K], trial_assembly_gpu_indexes[1, L]] += real(R1) * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[1, L]
    @atomic result[2, test_assembly_gpu_indexes[1, K], trial_assembly_gpu_indexes[1, L]] += imag(R1) * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[1, L]
    
    @atomic result[1, test_assembly_gpu_indexes[1, K], trial_assembly_gpu_indexes[2, L]] += real(R2) * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[2, L]
    @atomic result[2, test_assembly_gpu_indexes[1, K], trial_assembly_gpu_indexes[2, L]] += imag(R2) * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[2, L]
    
    @atomic result[1, test_assembly_gpu_indexes[1, K], trial_assembly_gpu_indexes[3, L]] += real(R3) * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[3, L]
    @atomic result[2, test_assembly_gpu_indexes[1, K], trial_assembly_gpu_indexes[3, L]] += imag(R3) * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[3, L]
    
    @atomic result[1, test_assembly_gpu_indexes[2, K], trial_assembly_gpu_indexes[1, L]] += real(R4) * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[1, L]
    @atomic result[2, test_assembly_gpu_indexes[2, K], trial_assembly_gpu_indexes[1, L]] += imag(R4) * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[1, L]
    
    @atomic result[1, test_assembly_gpu_indexes[2, K], trial_assembly_gpu_indexes[2, L]] += real(R5) * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[2, L]
    @atomic result[2, test_assembly_gpu_indexes[2, K], trial_assembly_gpu_indexes[2, L]] += imag(R5) * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[2, L]
    
    @atomic result[1, test_assembly_gpu_indexes[2, K], trial_assembly_gpu_indexes[3, L]] += real(R6) * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[3, L]
    @atomic result[2, test_assembly_gpu_indexes[2, K], trial_assembly_gpu_indexes[3, L]] += imag(R6) * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[3, L]
    
    @atomic result[1, test_assembly_gpu_indexes[3, K], trial_assembly_gpu_indexes[1, L]] += real(R7) * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[1, L]
    @atomic result[2, test_assembly_gpu_indexes[3, K], trial_assembly_gpu_indexes[1, L]] += imag(R7) * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[1, L]
    
    @atomic result[1, test_assembly_gpu_indexes[3, K], trial_assembly_gpu_indexes[2, L]] += real(R8) * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[2, L]
    @atomic result[2, test_assembly_gpu_indexes[3, K], trial_assembly_gpu_indexes[2, L]] += imag(R8) * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[2, L]
    
    @atomic result[1, test_assembly_gpu_indexes[3, K], trial_assembly_gpu_indexes[3, L]] += real(R9) * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[3, L]
    @atomic result[2, test_assembly_gpu_indexes[3, K], trial_assembly_gpu_indexes[3, L]] += imag(R9) * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[3, L]

end

@inline function store_with_kernel_register!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, K, L, x_offset, y_offset, P, T::GpuWriteBackTrue)
    @unroll for k in 1:9
        remainder, quotient = divrem(k - 1, 3)
        remainder += 1
        quotient  += 1

        @atomic result[1, test_assembly_gpu_indexes[remainder, K], trial_assembly_gpu_indexes[quotient, L]] += real(P[k]) * test_assembly_gpu_values[remainder, K] * trial_assembly_gpu_values[quotient, L]
        @atomic result[2, test_assembly_gpu_indexes[remainder, K], trial_assembly_gpu_indexes[quotient, L]] += imag(P[k]) * test_assembly_gpu_values[remainder, K] * trial_assembly_gpu_values[quotient, L]
    end
end

@inline function store_with_kernel_register!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, K, L, x_offset, y_offset, R1,R2,R3,R4,R5,R6,R7,R8,R9, T::GpuWriteBackFalse)
    k = K - x_offset
    l = L - y_offset
    result[k, l, 1] = R1 * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[1, L]
    result[k, l, 2] = R2 * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[2, L]
    result[k, l, 3] = R3 * test_assembly_gpu_values[1, K] * trial_assembly_gpu_values[3, L]
    result[k, l, 4] = R4 * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[1, L]
    result[k, l, 5] = R5 * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[2, L]
    result[k, l, 6] = R6 * test_assembly_gpu_values[2, K] * trial_assembly_gpu_values[3, L]
    result[k, l, 7] = R7 * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[1, L]
    result[k, l, 8] = R8 * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[2, L]
    result[k, l, 9] = R9 * test_assembly_gpu_values[3, K] * trial_assembly_gpu_values[3, L]
end

@inline function store_with_kernel_register!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, K, L, x_offset, y_offset, P, T::GpuWriteBackFalse)
    k = K - x_offset
    l = L - y_offset
    @unroll for i in 1:3
        @unroll for j in 1:3
            result[k, l, (i - 1)*3 + j] = P[(i - 1)*3 + j] * test_assembly_gpu_values[i, K] * trial_assembly_gpu_values[j, L]
        end
    end
end

# function store_with_kernel_register!(result, test_assembly_gpu, trial_assembly_gpu, K, L, R1,R2,R3,R4,R5,R6,R7,R8,R9, T::GpuWriteBackFalse)
#     result[1, K, L] = R1
#     result[2, K, L] = R2
#     result[3, K, L] = R3
#     result[4, K, L] = R4
#     result[5, K, L] = R5
#     result[6, K, L] = R6
#     result[7, K, L] = R7
#     result[8, K, L] = R8
#     result[9, K, L] = R9
# end


# function store_with_kernel_register!(result, test_assembly_gpu, trial_assembly_gpu, K, L, R1,R2,R3,R4,R5,R6,R7,R8,R9, T::GpuWriteBackFalse)
#     result[1, K, L, 1] = real(R1)
#     result[2, K, L, 1] = imag(R1)
#     result[1, K, L, 2] = real(R2)
#     result[2, K, L, 2] = imag(R2)
#     result[1, K, L, 3] = real(R3)
#     result[2, K, L, 3] = imag(R3)
#     result[1, K, L, 4] = real(R4)
#     result[2, K, L, 4] = imag(R4)
#     result[1, K, L, 5] = real(R5)
#     result[2, K, L, 5] = imag(R5)
#     result[1, K, L, 6] = real(R6)
#     result[2, K, L, 6] = imag(R6)
#     result[1, K, L, 7] = real(R7)
#     result[2, K, L, 7] = imag(R7)
#     result[1, K, L, 8] = real(R8)
#     result[2, K, L, 8] = imag(R8)
#     result[1, K, L, 9] = real(R9)
#     result[2, K, L, 9] = imag(R9)
# end