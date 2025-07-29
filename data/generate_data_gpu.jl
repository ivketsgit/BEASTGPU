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
            "data/GPU/$(inv_density_factor)/sortCPU"
        )

    M = assemble_gpu(S,X,X,config,config.writeBackStrategy)
    GC.gc()

    # config = GPUConfiguration(
    #         GpuWriteBackTrueInstance(),
    #         1,
    #         3 * GiB,
    #         doubleQuadRuleGpuStrategyShouldCalculateInstance(),
    #         ShouldCalcTrueInstance(),
    #         24 * GiB,
    #         16,
    #         true,
    #         CUDABackend(),
    #         TimeLogger(),
    #         Float64,
    #         false,
    #         "data/GPU/$(inv_density_factor)/sortGPU"
    #     )

    # M = assemble_gpu(S,X,X,config,config.writeBackStrategy)
    # GC.gc()


end

nothing