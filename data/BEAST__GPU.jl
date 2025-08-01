using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization

include("graph_data.jl")

const MiB = 2^20
const GiB = 2^30
CUDA.allowscalar(false)
config = GPUConfiguration(
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
        ""
    )


storeStrategies = ["storeGPU", "storeCPU"]
sortStrategies = ["sortGPU", "sortCPU"]


samples = 100
for inv_density_factor in density_values
    Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
    X = lagrangec0d1(Γ) 
    S = Helmholtz3D.singlelayer(wavenumber = 1.0)


    for inv_density_factor in density_values
    times = @benchmark assemble_gpu($S,$X,$X,$config,$config.writeBackStrategy) samples=samples evals=1 seconds=3600 * 2
    
    
    num_runs = length(times.times)
    total_duration = sum(times.times) / 1e9
    open("data/GPU_full/$(inv_density_factor)/sortGPU/storeGPU/full_time.txt", "a") do file
        println(file, """
            Manual Benchmark of duration $(total_duration) seconds over $(num_runs) runs:"
            $(times.times / 1e9)
            """
        )
    end
end