using CUDA
using CUDA.CUDAKernels

include("log.jl")
include("../CustomDataStructs/doubleQuadRuleGpuStrategy.jl")
include("../CustomDataStructs/GpuWriteBack.jl")
include("../CustomDataStructs/SauterSchwabQuadratureDataStruct.jl")
include("../CustomDataStructs/should_calc.jl")

# Define the config type
struct GPUConfiguration
    writeBackStrategy::Any
    amount_of_gpus::Int
    total_GPU_budget::Int
    InstancedoubleQuadRuleGpuStrategyShouldCalculate::Any
    ShouldCalcInstance::Any
    GPU_budget_pipeline_result::Int
    amount_of_producers::Int
    makeCompexWithGPU::Bool
    backend::CUDABackend
    timeLogger::TimeLogger
    floatType::Type
    sortOnCPU::Bool
    filename_benchmark::String
end

# Helper constructor function
function GPUConfiguration()
    CUDA.allowscalar(false)  # Important CUDA setting

    MiB = 2^20
    GiB = 2^30

    return GPUConfiguration(
        GpuWriteBackTrueInstance(),
        1,
        3 * GiB,
        doubleQuadRuleGpuStrategyShouldCalculateInstance(),
        ShouldCalcTrueInstance(),
        24 * GiB,
        16,
        true,
        CUDABackend(),
        TimeLogger(),
        Float64,
        false,
        filename_benchmark = ""
    )
end

