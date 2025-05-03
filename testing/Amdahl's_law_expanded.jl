using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization
using Logging

global_logger(ConsoleLogger(stderr, Logging.Warn))


include("../assamble_gpu.jl")
include("../CustomDataStructs/GpuWriteBack.jl")
include("../utils/backend.jl")



writeBackStrategy = GpuWriteBackTrueInstance()
#warmup
# Γ = meshcuboid(1.0,1.0,1.0,1.0)
# dimension(Γ)
# X = lagrangec0d1(Γ) 
# S = Helmholtz3D.singlelayer(wavenumber = 1.0)
# M_ref = BEAST.assemble(S,X,X)
# M_ref_gpu = assemble_gpu(S,X,X,writeBackStrategy,1)

# const B = 8
const MiB = 2^20
const GiB = 2^30
configuration = Dict()
configuration["writeBackStrategy"] = writeBackStrategy
configuration["amount_of_gpus"] = 1
configuration["total_GPU_budget"] = 3 * GiB
configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"] = doubleQuadRuleGpuStrategyShouldCalculateInstance()
configuration["ShouldCalcInstance"] = ShouldCalcTrueInstance()
configuration["GPU_budget_pipeline_result"] = 12 * GiB

inv_density_factor = 40
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
filename = "cashed_results/matrix_ref_$inv_density_factor.bin"

# a = @benchmark BEAST.assemble($S,$X,$X)
# display(a)
assemble_gpu(S,X,X,configuration)
b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=20 evals=1
display(b)

b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
@show b_stats

open("testing/results.txt", "a") do io
    write(io, string(inv_density_factor, configuration, " ", string(b_stats)))
end

