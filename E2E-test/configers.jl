using CUDA
using CUDA.CUDAKernels

include("../utils/log.jl")
include("../utils/configuration.jl")
function generate_configs()
    configs = []
    MiB = 2^20
    GiB = 2^30
    CUDA.allowscalar(false)



    writeBackStrategies = [ GpuWriteBackFalseInstance(), GpuWriteBackTrueInstance(),]
    amount_of_gpus = [1, 2]
    total_GPU_budgets = [3 * GiB]
    InstancedoubleQuadRuleGpuStrategyShouldCalculates = [doubleQuadRuleGpuStrategyShouldCalculateInstance()]
    ShouldCalcInstances = [ShouldCalcTrueInstance(), ShouldCalcFalseInstance()]
    GPU_budget_pipeline_results = [24 * GiB]
    makeCompexWithGPUs = [true, false]
    for writeBackStrategy in writeBackStrategies
        for amount_of_gpu in amount_of_gpus
            for total_GPU_budget in total_GPU_budgets
                for InstancedoubleQuadRuleGpuStrategyShouldCalculate in InstancedoubleQuadRuleGpuStrategyShouldCalculates
                    for ShouldCalcInstance in ShouldCalcInstances
                        for GPU_budget_pipeline_result in GPU_budget_pipeline_results
                            for makeCompexWithGPU in makeCompexWithGPUs
                                if writeBackStrategy isa GpuWriteBackFalseInstance && makeCompexWithGPUs == true
                                    break
                                end
                                config = GPUConfiguration(
                                    writeBackStrategy,
                                    amount_of_gpu,
                                    total_GPU_budget,
                                    InstancedoubleQuadRuleGpuStrategyShouldCalculate,
                                    ShouldCalcInstance,
                                    GPU_budget_pipeline_result,
                                    16,
                                    makeCompexWithGPU,
                                    CUDABackend(),
                                    TimeLogger(),
                                    Float64
                                )
                                push!(configs, config)
                            end
                        end 
                    end 
                end 
            end 
        end 
    end
    return configs
end



function print_fields(config::GPUConfiguration)
    to_print = Symbol[
        :writeBackStrategy,
        :amount_of_gpus,
        # :InstancedoubleQuadRuleGpuStrategyShouldCalculate,
        :ShouldCalcInstance,
        :makeCompexWithGPU,
    ]
    for field in fieldnames(GPUConfiguration)
        if field in to_print
            value = getfield(config, field)
            println("$(field), $(value)")
        end
    end
end
