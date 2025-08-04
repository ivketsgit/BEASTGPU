using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization

include("graph_data.jl")
include("../utils/configuration.jl")
include("../assamble_gpu.jl")

const MiB = 2^20
const GiB = 2^30
CUDA.allowscalar(false)


Γ = meshcuboid(1.0,1.0,1.0,0.5/40)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)


samples = 100
for threads in partial_store_threads
    file_name = "data/GPU/pStore_threads/time_threads_$(threads).txt"
    config = GPUConfiguration(
        GpuWriteBackFalseInstance(),
        1,
        3 * GiB,
        doubleQuadRuleGpuStrategyShouldCalculateInstance(),
        ShouldCalcTrueInstance(),
        24 * GiB,
        threads,
        true, #make complex with GPU
        CUDABackend(),
        TimeLogger(),
        Float64,
        false,
        "",
        file_name
    )
    
    open(file_name, "a") do file
        print(file, """Manual Benchmark over $(samples) runs:
        [""")
    end

    @benchmark assemble_gpu($S,$X,$X,$config,$config.writeBackStrategy) samples=samples evals=1 seconds=3600

    open(file_name, "a") do file
        print(file, "]")
    end 
end
