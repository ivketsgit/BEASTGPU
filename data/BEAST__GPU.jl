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



storeStrategies = ["storeGPU", "storeCPU"]
storeStrategies_ = [GpuWriteBackTrueInstance(), GpuWriteBackFalseInstance()]
sortStrategies = ["sortGPU", "sortCPU"]
sortStrategies_ = [true, false]


samples = 100


for inv_density_factor in density_values
    Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
    X = lagrangec0d1(Γ) 
    S = Helmholtz3D.singlelayer(wavenumber = 1.0)

    for (storeStrategy, storeStrategy_) in zip(storeStrategies, storeStrategies_)
        for (sortStrategy, sortStrategy_) in zip(sortStrategies, sortStrategies_)
            config = GPUConfiguration(
                storeStrategy_, #GpuWriteBackTrueInstance(),
                1,
                3 * GiB,
                doubleQuadRuleGpuStrategyShouldCalculateInstance(),
                ShouldCalcTrueInstance(),
                24 * GiB,
                16,
                true, #make complex with GPU
                CUDABackend(),
                TimeLogger(),
                Float64,
                sortStrategy_, #sort on CPU
                ""
            )
            # for inv_density_factor in density_values
            times = @benchmark assemble_gpu($S,$X,$X,$config,$config.writeBackStrategy) samples=samples evals=1 seconds=3600 * 2
            
            
            num_runs = length(times.times)
            total_duration = sum(times.times) / 1e9
            open("data/GPU_full/$(sortStrategy)/$(storeStrategy)/$(inv_density_factor)/full_time.txt", "a") do file
                println(file, """
                    Manual Benchmark of duration $(total_duration) seconds over $(num_runs) runs:"
                    $(times.times / 1e9)
                    """
                )
            end
        end
    end
end