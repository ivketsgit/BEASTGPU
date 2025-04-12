include("strategy_sauterschwabints.jl") 
include("Integrand_gpu.jl")
include("gpu_reduce.jl")
include("store_with_kernel.jl")  
include("CustomDataStructs/SauterSchwabQuadratureDataStruct.jl")
include("pulledback_integrand_gpu.jl")
using .Strategies             

using KernelAbstractions
using KernelAbstractions: @atomic
using KernelAbstractions.Extras: @unroll
using BEAST, CompScienceMeshes
using StaticArrays

function print_array(arr)
    @print("\n[")
    dims = size(arr)
    for i in 1:dims[1]
        for j in 1:dims[2]
            @print(arr[i, j], ", ")
        end
        @print("; ")
    end
    @print("]")
end


function calc_index_bitshifts(number)
    h = ((number - 1) >>> 8) + 1
    offset = (number - 1) & 0xFF  # Extract remainder from 256
    Local_number_offset = offset

    i = (offset >>> 6) + 1
    offset &= 0x3F  # Equivalent to offset % 64

    j = (offset >>> 4) + 1
    offset &= 0xF  # Equivalent to offset % 16

    k = (offset >>> 2) + 1
    l = (offset & 0x3) + 1
    return h, i, j, k, l, Local_number_offset
end

# const warpsize = 32
function load_global_memory_into_shared_memory!(h, Local_number_offset, store_index_, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, vertices1, vertices2, tangents1, tangents2, qps_, test_vert_, trail_vert_, test_tan_, trail_tan_, test_vol_, trail_vol_, ichart1_vert_, ichart2_vert_, ichart1_tan_, ichart2_tan_)
    warpsize = 32
    p = store_index_[h, 1]
    q = store_index_[h, 2]
    # if h == 1 && Local_number_offset == 0
    #     @print("\n ", p, " ", q)
    # end
    if  Local_number_offset == 0 * warpsize
        @unroll for unroll in 1:4
            qps[unroll, 1] = qps_[h, unroll, 1]
            qps[unroll, 2] = qps_[h, unroll, 2]
        end
    elseif Local_number_offset == 1 * warpsize
        @unroll for unroll in 1:3
            @unroll for unroll_ in 1:2
                vertices1[unroll, unroll_] = ichart1_vert_[h, unroll, unroll_]
                vertices2[unroll, unroll_] = ichart2_vert_[h, unroll, unroll_]
            end
        end
    elseif Local_number_offset == 2 * warpsize
        @unroll for unroll in 1:2
            @unroll for unroll_ in 1:2
                tangents1[unroll, unroll_] = ichart1_tan_[h, unroll, unroll_]
                tangents2[unroll, unroll_] = ichart2_tan_[h, unroll, unroll_]
            end
        end
    elseif Local_number_offset == 3 * warpsize
        @unroll for unroll in 1:3
            @unroll for unroll_ in 1:3
                test_vert[unroll, unroll_] = test_vert_[p, unroll, unroll_]
                trail_vert[unroll, unroll_] = trail_vert_[q, unroll, unroll_]
            end
        end
    elseif Local_number_offset == 4 * warpsize
        @unroll for unroll in 1:3
            @unroll for unroll_ in 1:2
                test_tan[unroll, unroll_] = test_tan_[p, unroll, unroll_]
                trail_tan[unroll, unroll_] = trail_tan_[q, unroll, unroll_]
            end
        end
    elseif Local_number_offset == 5 * warpsize
        test_vol[1] = test_vol_[p]
        trail_vol[1] = trail_vol_[q]
    end

    # return qps, vertices1, vertices2, tangents1, tangents2, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol
end

function load_data_1(cpu_data_, length, length_1, length_2, backend)
    data = Array{Float64}(undef,length,length_1,length_2)
    for i in 1:length
        for j in 1:length_2
            for k in 1:length_1
                data[i,k,j] = cpu_data_[i][j][k]
            end
        end
    end
    @assert !any(isnan, data)
    return move(backend, data)
end

function load_data_2(cpu_data_, length, length_1, length_2, backend)
    data = Array{Float64}(undef,length,length_1,length_2)
    for i in 1:length
        for j in 1:length_1
            for k in 1:length_2
                data[i,j,k] = cpu_data_[i][j][k]
            end
        end
    end
    @assert !any(isnan, data)
    return data
end

function load_data_to_gpu(cpu_data, length, backend, T, test_vert, trail_vert) 
    qps = load_data_2(cpu_data.qps, length, 4, 2, backend)
    store_index = Matrix{Int64}(undef,length,2)
    
    for i in 1:length
        store_index[i,:] = cpu_data.store_index[i][:]
    end

    @assert !any(isnan, store_index)
    store_index = move(backend, store_index)


       
    if prod(size(cpu_data.ichart1_vert)) != 0
        ichart1_vert = load_data_2(cpu_data.ichart1_vert, length, 3, 2, backend)
        ichart2_vert = load_data_2(cpu_data.ichart2_vert, length, 3, 2, backend)
        ichart1_tan = load_data_2(cpu_data.ichart1_tan, length, 2, 2, backend)
        ichart2_tan = load_data_2(cpu_data.ichart2_tan, length, 2, 2, backend)
    else
        ichart1_vert = KernelAbstractions.zeros(backend, Float64, (length, 3, 2))
        ichart2_vert = KernelAbstractions.zeros(backend, Float64, (length, 3, 2))
        ichart1_tan = KernelAbstractions.zeros(backend, Float64, (length, 2, 2))
        ichart2_tan = KernelAbstractions.zeros(backend, Float64, (length, 2, 2))
        
        vertices1_ = move(backend, Matrix{Float64}([1.0 0.0; 0.0 1.0; 0.0 0.0]))
        vertices2_ = move(backend, Matrix{Float64}([1.0 0.0; 0.0 1.0; 0.0 0.0]))
    
        test_toll!(backend, 512)(ichart1_vert, ichart2_vert, ichart1_tan, ichart2_tan, vertices1_, vertices2_, store_index, test_vert, trail_vert, T, ndrange = length)
        KernelAbstractions.synchronize(backend)
    end

    

    return qps, store_index, ichart1_vert, ichart2_vert, ichart1_tan, ichart2_tan
end

@kernel function sauterschwab_parameterized_gpu_outside_loop_kernel!(result,
    @Const(qps_), 
    @Const(test_vert_), @Const(trail_vert_), @Const(test_tan_), @Const(trail_tan_), 
    @Const(test_vol_), @Const(trail_vol_), 
    @Const(ichart1_vert_), @Const(ichart2_vert_),  @Const(ichart1_tan_), @Const(ichart2_tan_), 
    @Const(store_index_), 
    @Const(test_assembly_gpu_indexes), @Const(trial_assembly_gpu_indexes), @Const(test_assembly_gpu_values), @Const(trial_assembly_gpu_values),
    @Const(γ), @Const(α), @Const(T::CustomGpuData), @Const(writeBackStrategy::GpuWriteBack))

    Global_number = @index(Global, Linear)
    h, i, j, k, l, Local_number_offset = calc_index_bitshifts(Global_number)
    
    igd_Integrands = @localmem Float64 (4*4*4*4* 9 * 2)
    
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

    load_global_memory_into_shared_memory!(h, Local_number_offset, store_index_, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, vertices1, vertices2, tangents1, tangents2, qps_, test_vert_, trail_vert_, test_tan_, trail_tan_, test_vol_, trail_vol_, ichart1_vert_, ichart2_vert_, ichart1_tan_, ichart2_tan_)
    
    @synchronize
    η1 = qps[i, 1]
    η2 = qps[j, 1]
    η3 = qps[k, 1]
    ξ =  qps[l, 1]
    w = qps[i, 2] * qps[j, 2] * qps[k, 2] * qps[l, 2]
    
    calculate_part_quadrature(Local_number_offset, igd_Integrands, vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, η1, η2, η3, ξ, w, T)

    reduce_attomic(igd_Integrands, Local_number_offset)

    if Local_number_offset == 0
        store_with_kernel_splits!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, igd_Integrands, store_index_[h, 1], store_index_[h, 2], writeBackStrategy, Global_number)
    end
end


function calculate_part_quadrature(Local_number_offset, igd_Integrands, vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, η1, η2, η3, ξ, w, T::SauterSchwabQuadratureCommonVertexCustomGpuData)
    ξη1 = ξ * η1
    ξη2 = ξ * η2

    mul_ = w * (ξ^3) * η2
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξη1), (1 - ξη2, ξη2 * η3), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξη2, ξη2 * η3), (1 - ξ, ξη1), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
end

function calculate_part_quadrature(Local_number_offset, igd_Integrands, vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, η1, η2, η3, ξ, w, T::SauterSchwabQuadratureCommonEdgeCustomGpuData)
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
end

function calculate_part_quadrature(Local_number_offset, igd_Integrands, vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, η1, η2, η3, ξ, w, T::SauterSchwabQuadratureCommonFaceCustomGpuData)        
    mul = w * (ξ^3) * ((η1)^2) * (η2)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), (1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), (1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), (1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), (1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), (1 - ξ, ξ * η1 * (1 - η2)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
    Integrand__mul_gpu_attomic!(Local_number_offset, igd_Integrands,(1 - ξ, ξ * η1 * (1 - η2)), (1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul)
end

function sauterschwab_parameterized_gpu_outside_loop!(SauterSchwabQuadratureCustomGpuData,
    test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, 
    biop, instances::CustomGpuData, writeBackStrategy::GpuWriteBack, time_table, index,
    test_vert, test_tan, test_vol,
    trail_vert, trail_tan, trail_vol,
    store, length_return_matrix, size_qrule, SauterSchwabQuadraturetype, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, time_to_store)


    time_1 = @elapsed begin
        # backend = KernelAbstractions.get_backend(result)
        length = size(SauterSchwabQuadratureCustomGpuData.store_index)[1]
        α = biop.alpha
        γ = biop.gamma
        qps, store_index, ichart1_vert, ichart2_vert, ichart1_tan, ichart2_tan = load_data_to_gpu(SauterSchwabQuadratureCustomGpuData, length, backend, instances, test_vert, trail_vert)
    end
    
    result = create_results_matrix_gpu(backend, length_return_matrix, size_qrule, writeBackStrategy, SauterSchwabQuadratureCustomGpuData)

    time_2 = @elapsed begin
        qps, store_index, ichart1_vert, ichart2_vert, ichart1_tan, ichart2_tan = move(backend, qps), move(backend, store_index), move(backend, ichart1_vert), move(backend, ichart2_vert), move(backend, ichart1_tan), move(backend, ichart2_tan)

        
        sauterschwab_parameterized_gpu_outside_loop_kernel!(backend, 256)(result, qps, 
            test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, ichart1_vert, ichart2_vert, ichart1_tan, ichart2_tan, store_index, 
            test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, 
            γ, α, instances, writeBackStrategy, ndrange = (4 * 4 * 4 * 4 * length))
        KernelAbstractions.synchronize(backend)
    end

    if typeof(writeBackStrategy) == GpuWriteBackFalseInstance
        time_to_store_ = @elapsed begin
            write_to_compact_matrix(result, store, length_return_matrix, size_qrule, writeBackStrategy, SauterSchwabQuadraturetype, test_assembly_cpu_indexes, trial_assembly_cpu_indexes)
        end
    end
    # Threads.atomic_add!(time_to_store, time_to_store_)

    Threads.atomic_add!(time_table[1,index], time_1)
    Threads.atomic_add!(time_table[2,index], time_2)
    # time_table[1,index] += time_1
    # time_table[2,index] += time_2
end