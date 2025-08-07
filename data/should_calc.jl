using Plots, Profile, FlameGraphs  


include("../CustomDataStructs/GpuWriteBack.jl")
include("../utils/configuration.jl")

include("../assamble_gpu.jl")
include("graph_data.jl")


const MiB = 2^20
const GiB = 2^30
CUDA.allowscalar(false)


for inv_density_factor in density_values
    Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
    X = lagrangec0d1(Γ) 
    S = Helmholtz3D.singlelayer(wavenumber = 1.0)

    
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
            true,
            "data/GPU/$(inv_density_factor)/should_calc_true.txt",
            nothing
        )
    while true
        try
            M = assemble_gpu(S, X, X, config, config.writeBackStrategy)
            break  # Exit loop if successful
        catch e
            println("Error occurred: ", e)
            # Optionally: sleep(1) to pause before retrying
        end
end
    GC.gc()
end

for inv_density_factor in density_values
    Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
    X = lagrangec0d1(Γ) 
    S = Helmholtz3D.singlelayer(wavenumber = 1.0)

    
    config = GPUConfiguration(
            GpuWriteBackTrueInstance(),
            1,
            3 * GiB,
            doubleQuadRuleGpuStrategyShouldCalculateInstance(),
            ShouldCalcFalseInstance(),
            24 * GiB,
            16,
            true,
            CUDABackend(),
            TimeLogger(),
            Float64,
            true,
            "data/GPU/$(inv_density_factor)/should_calc_false.txt",
            nothing
        )
    while true
        try
            M = assemble_gpu(S, X, X, config, config.writeBackStrategy)
            break  # Exit loop if successful
        catch e
            println("Error occurred: ", e)
            # Optionally: sleep(1) to pause before retrying
        end
    end
    GC.gc()
end

nothing