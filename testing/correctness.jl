using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization
using Logging

include("../assamble_gpu.jl")
include("../utils/backend.jl")

global_logger(ConsoleLogger(stderr, Logging.Warn))

inv_density_factor = 1
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)

filename = "cashed_results/matrix_ref_$inv_density_factor.bin"
M_ref = open(filename, "r") do io
    deserialize(io)
end

const MiB = 2^20
const GiB = 2^30
@show(eps(Float64))

config = Dict()

writeBackStrategy = [GpuWriteBackTrueInstance(), GpuWriteBackFalseInstance()]
amount_of_gpus = [1, 2]
total_GPU_budget = [3 * GiB]
InstancedoubleQuadRuleGpuStrategyShouldCalculate = [doubleQuadRuleGpuStrategyShouldCalculateInstance()]
ShouldCalcInstance = [ShouldCalcTrueInstance(), ShouldCalcFalseInstance()]
GPU_budget_pipeline_result = [6 * GiB]
for e1 in writeBackStrategy
    config["writeBackStrategy"] = e1
    for e2 in amount_of_gpus
        config["amount_of_gpus"] = e2
        for e3 in total_GPU_budget
            config["total_GPU_budget"] = e3
            for e4 in InstancedoubleQuadRuleGpuStrategyShouldCalculate
                config["InstancedoubleQuadRuleGpuStrategyShouldCalculate"] = e4
                for e5 in ShouldCalcInstance
                    config["ShouldCalcInstance"] = e5
                    for e6 in GPU_budget_pipeline_result
                        config["GPU_budget_pipeline_result"] = e6

                        @show config
                        # @show M_gpu config
                        @show maximum(abs.(M_ref .- assemble_gpu(S, X, X, config)))
                        println("")
                    end 
                end 
            end 
        end 
    end 
end

config = Dict{Any, Any}("total_GPU_budget" => 3221225472, 
"InstancedoubleQuadRuleGpuStrategyShouldCalculate" => doubleQuadRuleGpuStrategyShouldCalculateInstance(), 
"GPU_budget_pipeline_result" => 6442450944, "amount_of_gpus" => 2, "ShouldCalcInstance" => ShouldCalcFalseInstance(), 
"writeBackStrategy" => GpuWriteBackTrueInstance())
maximum(abs.(M_ref .- assemble_gpu(S, X, X, config))) = 1.3904866443919908e-17




# error_matrix = abs.(M_ref .- M_ref_gpu)
# @show maximum(error_matrix)