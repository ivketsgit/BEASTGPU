
function calc_index(number)
    h_, Local_number_offset = divrem(number - 1, 256)
    h = h_ + 1
    i_, quotient = divrem(Local_number_offset, 64)
    i = i_ + 1
    j_, quotient = divrem(quotient, 16)
    j = j_ + 1
    k_, l_ = divrem(quotient, 4)
    k = k_ + 1
    l = l_ + 1
    return h, i, j, k, l, Local_number_offset
end

@kernel function sauterschwab_parameterized_CommonVertex_gpu_outside_loop_kernel!(result,
    @Const(qps_), 
    @Const(test_vert_), @Const(trail_vert_), 
    @Const(test_tan_), @Const(trail_tan_), 
    @Const(test_vol_), @Const(trail_vol_), 
    @Const(ichart1_vert_), @Const(ichart2_vert_), 
    @Const(ichart1_tan_), @Const(ichart2_tan_), 
    @Const(store_index_),
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(γ), @Const(α))

    Global_number = @index(Global, Linear)
    h, i, j, k, l, Local_number_offset = calc_index_bitshifts(Global_number)
    
    igd_Integrands = @localmem Float64 (4*4*4*4* 9 * 2) # 4^4, 9 * 2
    
    @unroll for unroll in 1:18
        igd_Integrands[Local_number_offset * 9 * 2 + unroll] = 0
    end

    qps = @localmem Float64 (4,2)
    vertices1 = @localmem Float64 (3,2)
    vertices2 = @localmem Float64 (3,2)
    tangents1 = @localmem Float64 (2,2)
    tangents2 = @localmem Float64 (2,2)
    test_vert = @localmem Float64 (3,3)
    trail_vert = @localmem Float64 (3,3)
    test_tan = @localmem Float64 (3,2)
    trail_tan = @localmem Float64 (3,2)
    test_vol = @localmem Float64 (1)
    trail_vol = @localmem Float64 (1)

    load_global_memory_into_shared_memory!(h, Local_number_offset, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, vertices1, vertices2, tangents1, tangents2, qps_, test_vert_, trail_vert_, test_tan_, trail_tan_, test_vol_, trail_vol_, ichart1_vert_, ichart2_vert_, ichart1_tan_, ichart2_tan_)
    
    @synchronize

    η1 = qps[i, 1]
    η2 = qps[j, 1]
    η3 = qps[k, 1]
    ξ =  qps[l, 1]

    ξη1 = ξ * η1
    ξη2 = ξ * η2

    w = qps[i, 2] * qps[j, 2] * qps[k, 2] * qps[l, 2]


    mul_ = w * (ξ^3) * η2
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξη1), (1 - ξη2, ξη2 * η3), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξη2, ξη2 * η3), (1 - ξ, ξη1), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)

    reduce_attomic(igd_Integrands, Local_number_offset)

    if Local_number_offset == 0
        store_with_kernel_splits!(result, test_assembly_gpu, trial_assembly_gpu, igd_Integrands, store_index_[h, 1], store_index_[h, 2])
    end
end

@kernel function sauterschwab_parameterized_CommonEdge_gpu_outside_loop_kernel!(result,
    @Const(qps_), 
    @Const(test_vert_), @Const(trail_vert_), 
    @Const(test_tan_), @Const(trail_tan_), 
    @Const(test_vol_), @Const(trail_vol_), 
    @Const(ichart1_vert_), @Const(ichart2_vert_), 
    @Const(ichart1_tan_), @Const(ichart2_tan_), 
    @Const(store_index_),
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(γ), @Const(α))

    Global_number = @index(Global, Linear)
    h, i, j, k, l, Local_number_offset = calc_index_bitshifts(Global_number)
    
    igd_Integrands = @localmem Float64 (4*4*4*4* 9 * 2) # 4^4, 9 * 2
    
    @unroll for unroll in 1:18
        igd_Integrands[Local_number_offset * 9 * 2 + unroll] = 0
    end

    qps = @localmem Float64 (4,2)
    vertices1 = @localmem Float64 (3,2)
    vertices2 = @localmem Float64 (3,2)
    tangents1 = @localmem Float64 (2,2)
    tangents2 = @localmem Float64 (2,2)
    test_vert = @localmem Float64 (3,3)
    trail_vert = @localmem Float64 (3,3)
    test_tan = @localmem Float64 (3,2)
    trail_tan = @localmem Float64 (3,2)
    test_vol = @localmem Float64 (1)
    trail_vol = @localmem Float64 (1)

    load_global_memory_into_shared_memory!(h, Local_number_offset, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, vertices1, vertices2, tangents1, tangents2, qps_, test_vert_, trail_vert_, test_tan_, trail_tan_, test_vol_, trail_vol_, ichart1_vert_, ichart2_vert_, ichart1_tan_, ichart2_tan_)
    
    @synchronize

    η1 = qps[i, 1]
    η2 = qps[j, 1]
    η3 = qps[k, 1]
    ξ =  qps[l, 1]
    w = qps[i, 2] * qps[j, 2] * qps[k, 2] * qps[l, 2]

    ξη1 = ξ * η1
    η1η2 = η1 * η2
    η2η3 = η2 * η3
    η1η2η3 = η1η2 * η3

    mul_ = w * (ξ^3) * ((η1)^2) * (η2)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξη1), (1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), (1 - ξ, ξη1 * η2η3), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), (1 - ξ, ξη1), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ * (1 - η1η2η3), ξη1 * (1 - η2η3)), (1 - ξ, ξη1 * η2), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
    
    mul_ = w * (ξ^3) * ((η1)^2)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξη1 * η3), (1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)

    reduce_attomic(igd_Integrands, Local_number_offset)

    if Local_number_offset == 0
        store_with_kernel_splits!(result, test_assembly_gpu, trial_assembly_gpu, igd_Integrands, store_index_[h, 1], store_index_[h, 2])
    end
end

@kernel function sauterschwab_parameterized_CommonFace_gpu_outside_loop_kernel!(result,
    @Const(qps_), 
    @Const(test_vert_), @Const(trail_vert_), 
    @Const(test_tan_), @Const(trail_tan_), 
    @Const(test_vol_), @Const(trail_vol_), 
    @Const(ichart1_vert_), @Const(ichart2_vert_), 
    @Const(ichart1_tan_), @Const(ichart2_tan_), 
    @Const(store_index_),
    @Const(test_assembly_gpu), @Const(trial_assembly_gpu),
    @Const(γ), @Const(α))

    Global_number = @index(Global, Linear)
    h, i, j, k, l, Local_number_offset = calc_index_bitshifts(Global_number)
    
    igd_Integrands = @localmem Float64 (4*4*4*4* 9 * 2) # 4^4, 9 * 2
    
    @unroll for unroll in 1:18
        igd_Integrands[Local_number_offset * 9 * 2 + unroll] = 0
    end

    qps = @localmem Float64 (4,2)
    vertices1 = @localmem Float64 (3,2)
    vertices2 = @localmem Float64 (3,2)
    tangents1 = @localmem Float64 (2,2)
    tangents2 = @localmem Float64 (2,2)
    test_vert = @localmem Float64 (3,3)
    trail_vert = @localmem Float64 (3,3)
    test_tan = @localmem Float64 (3,2)
    trail_tan = @localmem Float64 (3,2)
    test_vol = @localmem Float64 (1)
    trail_vol = @localmem Float64 (1)

    load_global_memory_into_shared_memory!(h, Local_number_offset, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, vertices1, vertices2, tangents1, tangents2, qps_, test_vert_, trail_vert_, test_tan_, trail_tan_, test_vol_, trail_vol_, ichart1_vert_, ichart2_vert_, ichart1_tan_, ichart2_tan_)
    
    @synchronize

    η1 = qps[i, 1]
    η2 = qps[j, 1]
    η3 = qps[k, 1]
    ξ =  qps[l, 1]
    
    w = qps[i, 2] * qps[j, 2] * qps[k, 2] * qps[l, 2]

    # mul_ = w1 * w2 * w3 * w4 * (ξ^3) * ((η1)^2) * (η2)
    
    mul = w * (ξ^3) * ((η1)^2) * (η2)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), (1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), (1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), (1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), (1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), (1 - ξ, ξ * η1 * (1 - η2)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξ * η1 * (1 - η2)), (1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)

    reduce_attomic(igd_Integrands, Local_number_offset)

    if Local_number_offset == 0
        store_with_kernel_splits!(result, test_assembly_gpu, trial_assembly_gpu, igd_Integrands, store_index_[h, 1], store_index_[h, 2])
    end
end

function sauterschwab_parameterized_gpu_outside_loop!(result, 
    SauterSchwabQuadratureCustomGpuData,
    test_assembly_gpu, trial_assembly_gpu, biop, type)

    # @show length
    time_1 = @elapsed begin
        backend = KernelAbstractions.get_backend(result)
        length = size(SauterSchwabQuadratureCustomGpuData.store_index)[1]
        α = biop.alpha
        γ = biop.gamma
        qps, store_index, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, ichart1_vert, ichart2_vert, ichart1_tan, ichart2_tan = load_data_to_gpu(SauterSchwabQuadratureCustomGpuData, length, backend)
    end
    # println("time to conv vect to matrix = ",time_1)

    # temp_results = KernelAbstractions.zeros(backend, ComplexF64, 9, 4, 4, 4, 4, length)

    time_2 = @elapsed begin
        # if type == "CommonVertex"
        #     kernel! = sauterschwab_parameterized_CommonVertex_gpu_outside_loop_kernel!(backend, 256)
        # elseif type == "CommonEdge"
        #     kernel! = sauterschwab_parameterized_CommonEdge_gpu_outside_loop_kernel!(backend, 256)
        # elseif type == "CommonFace"
        #     kernel! = sauterschwab_parameterized_CommonFace_gpu_outside_loop_kernel!(backend, 256)
        # else
        #     throw("type sauterschwab_parameterized gpu not supported")
        # end
        # kernel!(result, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, ichart1_vert, ichart2_vert, ichart1_tan, ichart2_tan, store_index, test_assembly_gpu, trial_assembly_gpu, γ, α, ndrange = (4 * 4 * 4 * 4 *length))
        
        T = SauterSchwabQuadratureCommonVertexCustomGpuData_()
        kernel! = sauterschwab_parameterized_gpu_outside_loop_kernel!(backend, 256)
        kernel!(result, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, ichart1_vert, ichart2_vert, ichart1_tan, ichart2_tan, store_index, test_assembly_gpu, trial_assembly_gpu, γ, α, T, ndrange = (4 * 4 * 4 * 4 *length))
        KernelAbstractions.synchronize(backend)
    end

    return time_1, time_2
    # temp_results = sum(temp_results, dims=(2, 3, 4, 5))
    # @show Array(temp_results)[:,1,1,1,1,3]
    # KernelAbstractions.synchronize(backend)


    # kernel! = store_with_kernel_splits_!(backend)
    # kernel!(result_real, result_imag, test_assembly_gpu, trial_assembly_gpu, temp_results, store_index, ndrange = (length))
    # KernelAbstractions.synchronize(backend)
end

@kernel function store_with_kernel_splits_!(result_real, result_imag, @Const(test_assembly_gpu), @Const(trial_assembly_gpu), @Const(igd_Integrands), @Const(store_index))
    i =  @index(Global, NTuple)[1]
    K = store_index[i,1]
    L = store_index[i,2]
    # if i == 3
        for k in 0:8
            remainder, quotient = divrem(k, 3)
            # @print("\n quotient = ", quotient, " remainder = ",remainder, " K = ",  K)
            # @print("\n igd = ", k+1, " ", i)
            # @print("\n igd = ", real(igd_Integrands[k+1,1,1,1,1,i]))


            # @print("\n i = ", i , " test_assembly_gpu[quotient + 1, K] = ",test_assembly_gpu[quotient + 1, K], " trial_assembly_gpu[remainder + 1, L] = ", trial_assembly_gpu[remainder + 1, L], " igd = ", real(igd_Integrands[k+1,1,1,1,1,i]) * 10^5, " ", imag(igd_Integrands[k+1,1,1,1,1,i]) * 10^5, "im")
            @print("\n quotient = ", quotient, " remainder = ",remainder, "   K = ",  K, " L = ", L, " igd_Integrands[k * 2 + 1] = ",igd_Integrands[k * 2 + 1] , " igd_Integrands[k * 2 + 2] = ", igd_Integrands[k * 2 + 2])
            @atomic result_real[test_assembly_gpu[quotient + 1, K], trial_assembly_gpu[remainder + 1, L]] += real(igd_Integrands[k+1,1,1,1,1,i])
            @atomic result_imag[test_assembly_gpu[quotient + 1, K], trial_assembly_gpu[remainder + 1, L]] += imag(igd_Integrands[k+1,1,1,1,1,i])
        end
    # end
end
