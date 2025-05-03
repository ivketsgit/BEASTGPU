include("SauterSchwabQuadratureDataStruct.jl")
include("doubleQuadRuleGpuStrategy.jl")
abstract type GpuWriteBack end

abstract type GpuWriteBackTrue <: GpuWriteBack end
abstract type GpuWriteBackFalse <: GpuWriteBack end

struct GpuWriteBackTrueInstance <: GpuWriteBackTrue end
struct GpuWriteBackFalseInstance <: GpuWriteBackFalse end

const gpu_results_cache = IdDict()
function create_results_matrix_gpu(backend, length_return_matrix, size_qrule, T::GpuWriteBackTrue, any)
    if haskey(gpu_results_cache, 1)
        return gpu_results_cache[1]
    else
        gpu_results_cache[1] = KernelAbstractions.zeros(backend, Float64, 2, length_return_matrix, length_return_matrix)
        return gpu_results_cache[1]
    end
    # return KernelAbstractions.zeros(backend, Float64, 2, length_return_matrix, length_return_matrix)
end

function create_results_matrix_gpu(backend, length_return_matrix, elements_length_tuple, T::GpuWriteBackFalse, T2::doubleQuadRuleGpuStrategy)
    return KernelAbstractions.allocate(backend, ComplexF64, elements_length_tuple[1], elements_length_tuple[2], 9)
    # return KernelAbstractions.zeros(backend, ComplexF64, size_qrule, size_qrule, 9)
end

function create_results_matrix_gpu(backend, length_return_matrix, size_qrule, T::GpuWriteBackFalse, T2::SauterSchwabQuadrature_gpu_data)
    return KernelAbstractions.allocate(backend, ComplexF64, size(T2.store_index)[1], 9)
    # return KernelAbstractions.zeros(backend, ComplexF64, size(T2.store_index)[1], 9)
end

function write_to_compact_matrix(gpu_matrix, store, length_return_matrix, ndrange, T::GpuWriteBackTrue, T2, trial_assembly_data, test_assembly_data)
    # result_cpu = Array(gpu_matrix)
    # result_cpu = complex.(view(result_cpu, 1, :, :), view(result_cpu, 2, :, :))

    # for i in 1:length_return_matrix
    #     for j in 1:length_return_matrix
    #         store(result_cpu[i,j], i, j)
    #     end
    # end
end

# function write_to_compact_matrix(gpu_matrix, store, length_return_matrix, size_qrule, T::GpuWriteBackFalse, T2::doubleQuadRuleGpuStrategy, trial_assembly_data, test_assembly_data)
#     result_cpu = Array(gpu_matrix)

#     @inline for q in 1:size_qrule
#         n_ = @view trial_assembly_data[:, q]
#         for j in 1 : 3
#             n = n_[j]
#             for p in 1:size_qrule
#                 z = @view result_cpu[p, q, :]
#                 m_ = @view test_assembly_data[:, p]
#                 for i in 1 : 3
#                     m = m_[i]
#                     zij = z[(i-1) * 3 + j]
#                     store(zij, m, n)
#                 end
#             end
#         end                 
#     end
# end

function write_to_compact_matrix(result_cpu, store, length_return_matrix, ndrange, T::GpuWriteBackFalse, T2::doubleQuadRuleGpuStrategy, test_assembly_data, trial_assembly_data, offsets)

    @inline for q in 1:ndrange[2]
        n_ = @view trial_assembly_data[:, q + offsets[2]]
        for j in 1 : 3
            n = n_[j]
            for p in 1:ndrange[1]
                z = @view result_cpu[p, q, :]
                m_ = @view test_assembly_data[:, p + offsets[1]]
                for i in 1 : 3
                    m = m_[i]
                    zij = z[(i-1) * 3 + j]
                    # @show zij, m, n
                    # if m * n != 0
                        store(zij, m, n)
                    # end
                end
            end
        end                 
    end
end

function write_to_compact_matrix(gpu_matrix, store, length_return_matrix, ndrange, T::GpuWriteBackFalse, T2::SauterSchwabQuadrature_gpu_data, test_assembly_data, trial_assembly_data)
    result_cpu = Array(gpu_matrix)

    for (iterator, (J, I)) in  enumerate(T2.store_index)
        # if iterator == 1
            z = @view result_cpu[iterator, :]
            for j in 1 : 3
                n = trial_assembly_data[j, I]
                for i in 1 : 3
                    m = test_assembly_data[i, J]
                    zij = z[(j-1) * 3 + i]
                    # @show i, j, zij, m, n, J, I
                    if m * n != 0
                        store(zij, m, n)
                    end
                end
            end
        # end
    end
end

@inline function get_zij(z, iterator, i, j, T::SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonFaceCustomGpuData})
    z[(i-1) * 3 + j]
end
@inline function get_zij(z, iterator, i, j, T::SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonEdgeCustomGpuData})
    z[(j-1) * 3 + i]
end
@inline function get_zij(z, iterator, i, j, T::SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonVertexCustomGpuData})
    z[(j-1) * 3 + i]
end