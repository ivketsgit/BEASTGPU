include("CustomDataStructs/doubleQuadRuleGpuStrategy.jl")
include("CustomDataStructs/should_calc.jl")

using KernelAbstractions, Atomix
using BEAST, CompScienceMeshes
using KernelAbstractions.Extras: @unroll
using KernelAbstractions: @atomic
using StaticArrays
include("store_with_kernel.jl")  

@inline function doubleQuadRuleMainCase!(result,
    elementAssemblyData,
    biop,
    wimps_and_womps, 
    offset, ndrange_,
    timingInfo, index,
    config,
    quadrule_types_gpu=0,
    SauterSchwabQuadratureCommonVertex=0, SauterSchwabQuadratureCommonEdge=0,SauterSchwabQuadratureCommonFace=0)

    instance = config.InstancedoubleQuadRuleGpuStrategyShouldCalculate
    writeBackStrategy = config.writeBackStrategy
    ShouldCalcInstance = config.ShouldCalcInstance

    backend = config.backend
    floatType = config.floatType
    store_index = load_data_(instance, backend, timingInfo.time_table, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)

    α = biop.alpha
    γ = biop.gamma

    elements_data = elementAssemblyData.elements_data
    assembly_gpu_data = elementAssemblyData.assembly_data
    

    time_1 = @elapsed begin
        task = Threads.@spawn begin
            doubleQuadRuleMainCaseKernel!(backend)(
                result,
                assembly_gpu_data[1:4]..., #test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values,
                γ, α,
                wimps_and_womps..., #womps_weights, womps_values, womps_cart, wimps_weights, wimps_values, wimps_cart,
                offset..., #x_offset, y_offset,
                instance, writeBackStrategy,
                quadrule_types_gpu,
                store_index,
                ShouldCalcInstance,
                elements_data[[1, 4, 6]]..., #test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix,
                floatmax(floatType),
                ndrange = ndrange_)
            KernelAbstractions.synchronize(backend)
        end
        wait(task)
    end
    Threads.atomic_add!(timingInfo.time_table[2,index], time_1)
    # time_table[2,index] += time_1
end


@kernel function doubleQuadRuleMainCaseKernel!(result,
    @Const(test_assembly_gpu_indexes), @Const(trial_assembly_gpu_indexes), @Const(test_assembly_gpu_values), @Const(trial_assembly_gpu_values),
    @Const(γ), @Const(α),
    @Const(womps_weights), @Const(womps_values), @Const(womps_cart),
    @Const(wimps_weights), @Const(wimps_values), @Const(wimps_cart), 
    @Const(x_offset), @Const(y_offset),
    @Const(T::doubleQuadRuleGpuStrategy), @Const(T2::GpuWriteBack), 
    @Const(quadrule_types_gpu), @Const(store_index),
    @Const(ShouldCalcInstance::ShouldCalc), @Const(test_elements_vertices_matrix), @Const(trial_elements_vertices_matrix), @Const(trial_elements_volume_matrix), @Const(floatmax_type))    

    K, L = @index(Global, NTuple)
    # K = get_index(T, K, L, store_index, true) + x_offset
    # L = get_index(T, K, L, store_index, false) + y_offset
    K = K + x_offset
    L = L + y_offset
    
    P = @private ComplexF64 (9)
    
    @inbounds begin
        for i in 1:9
            P[i] = 0.0
        end
        
        for I in 1:3
            for J in 1:4
                R = sqrt((womps_cart[K, I, 1] - wimps_cart[L, J, 1])^2 + 
                        (womps_cart[K, I, 2] - wimps_cart[L, J, 2])^2 + 
                        (womps_cart[K, I, 3] - wimps_cart[L, J, 3])^2 )
                

                j_αG = calc_j_αG(α, womps_weights[K, I], wimps_weights[L, J], R, γ, T, quadrule_types_gpu, K, L, ShouldCalcInstance, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatmax_type)

                @unroll for i in 0:2
                    j_αG_womps_values = j_αG * womps_values[K, I, i+1]
                    @unroll for j in 1:3
                        P[i * 3 + j] += j_αG_womps_values * wimps_values[L, J, j]
                    end
                end
            end
        end
        store_with_kernel_register!(result, test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, K, L, x_offset, y_offset, P, T2)
    end
end

function load_data_(T::doubleQuadRuleGpuStrategyRepaire, backend, time_table, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
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

        store_index = move(backend, store_index)
    end
    time_table[2,2] += time
    
    return store_index
end

function load_data_(T::doubleQuadRuleGpuStrategyOptimistic, backend, time_table, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace) 
    return 0
end
function load_data_(T::doubleQuadRuleGpuStrategyShouldCalculate, backend, time_table, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace) 
    return 0
end

const i4pi = 1 / (4pi)
const epsilon =  eps(Float64)
@inline function calc_j_αG(α, womps_weight, wimps_weight, R, γ, T::doubleQuadRuleGpuStrategyOptimistic, quadrule_types_gpu, K, L, ShouldCalcInstance, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatType)
    return α * womps_weight * wimps_weight * i4pi / R * exp(-R*γ)
end
@inline function calc_j_αG(α, womps_weight, wimps_weight, R, γ, T::doubleQuadRuleGpuStrategyRepaire, quadrule_types_gpu, K, L, ShouldCalcInstance, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatType)
    return - α * womps_weight * wimps_weight * i4pi / R * exp(-R*γ)
end
@inline function calc_j_αG(α, womps_weight, wimps_weight, R, γ, T::doubleQuadRuleGpuStrategyShouldCalculate, quadrule_types_gpu, K, L, ShouldCalcInstance::ShouldCalcTrue, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatType)
    # K_offset = K - 1
    
    return α * womps_weight * wimps_weight * i4pi / R * exp(-R*γ) * (quadrule_types_gpu[K, L] == 0)
    # return α * womps_weight * wimps_weight * i4pi / R * exp(-R*γ) * ((should_calc[(K_offset >> 5) + 1, L] >> ((K_offset & 31 ))) & 1)
end

@inline function calc_j_αG(α, womps_weight, wimps_weight, R, γ, T::doubleQuadRuleGpuStrategyShouldCalculate, should_calc, K, L, ShouldCalcInstance::ShouldCalcFalse, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatmax_type)
    K_offset = K - 1

    abs2_ = abs2(γ)
    should_calc_ = quadrule_determine_if_doubleQuadRule(K, L, abs2_, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatmax_type)
    
    return α * womps_weight * wimps_weight * i4pi / R * exp(-R*γ) * should_calc_
end

@inline function get_index(T::doubleQuadRuleGpuStrategyRepaire, K, L, store_index, bool)
    return store_index[K, bool + 1]
end


@inline function get_index(T::doubleQuadRuleGpuStrategyOptimistic, K, L, store_index, bool)
    return bool ? K : L
end
@inline function get_index(T::doubleQuadRuleGpuStrategyShouldCalculate, K, L, store_index, bool)
    return bool ? K : L
end


const dtol = 2.220446049250313e-16 * 1.0e3
const xtol2 = 0.2 * 0.2
@inline function quadrule_determine_if_doubleQuadRule(i,j, k2::Float64, τ::AbstractArray, σ::AbstractArray, σ_volume::AbstractArray, floatmax_type::Float64)
    hits = 0
    dmin2 = floatmax_type 
    @inbounds @unroll for unroll in 1:3
        @inbounds @unroll for unroll_ in 1:3
            d2 = (τ[i, 1, unroll] - σ[j, 1, unroll_])^2 + 
                 (τ[i, 2, unroll] - σ[j, 2, unroll_])^2 + 
                 (τ[i, 3, unroll] - σ[j, 3, unroll_])^2
            d = sqrt(d2)
            hits += (d < dtol)
            dmin2 = min(dmin2, d2)
        end
    end
    return (hits == 0 && max(dmin2*k2, dmin2/(16 * σ_volume[j])) >= xtol2 ? 1 : 0)
end