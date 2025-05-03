using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization
using Logging

# global_logger(ConsoleLogger(stderr, Logging.Warn))


include("../assamble_gpu.jl")
include("../CustomDataStructs/GpuWriteBack.jl")
include("../utils/backend.jl")



global_logger(ConsoleLogger(stderr, Logging.Warn))

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

#warmup
begin
    Γ_ = meshcuboid(1.0,1.0,1.0,0.5/1)
    X_ = lagrangec0d1(Γ_) 
    S_ = Helmholtz3D.singlelayer(wavenumber = 1.0)
    assemble_gpu(S_,X_,X_,configuration)
end

samples_ = 200
open("testing/results_atomic_vs_non_atomic.txt", "w") do file
    redirect_stdout(file) do
        print("\n{")
    end
end

for inv_density_factor in [15, 25, 30, 33, 35, 37, 39, 40]
    open("testing/results_atomic_vs_non_atomic.txt", "a") do file
        redirect_stdout(file) do
            print("\n ", inv_density_factor, " : [")
            Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
            X = lagrangec0d1(Γ) 
            S = Helmholtz3D.singlelayer(wavenumber = 1.0)
            b  = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples_ evals=1 seconds=9999999999999999999999999999999999999999999999999999999999999999999
            
            b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
            
            print(" ],")
        end
    end
    # open("testing/results.txt", "a") do io
    #     write(io, string("\n",inv_density_factor,  " ", samples, " ", configuration, " ", string(b_stats)))
    # end
end
open("testing/results_atomic_vs_non_atomic.txt", "w") do file
    redirect_stdout(file) do
        print("\a}")
    end
end        