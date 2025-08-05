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



config = GPUConfiguration(
        GpuWriteBackFalseInstance(),
        1,
        3 * GiB,
        doubleQuadRuleGpuStrategyShouldCalculateInstance(),
        ShouldCalcTrueInstance(),
        18 * GiB,
        16,
        true, #make complex with GPU
        CUDABackend(),
        TimeLogger(),
        Float64,
        false,
        "",
        nothing
    )

samples = 100


for inv_density_factor in density_values
    Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
    X = lagrangec0d1(Γ) 
    S = Helmholtz3D.singlelayer(wavenumber = 1.0)

        

    f = function()
        assemble_gpu(S,X,X,config,config.writeBackStrategy)
    end
    filename = "data/GPU_full/storeCPU/$(inv_density_factor)/full_time.txt"
    manual_benchmark(f, n=100,filename=filename, appendOrWrite="a")
end