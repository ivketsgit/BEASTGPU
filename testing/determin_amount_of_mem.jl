using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization
using Logging

global_logger(ConsoleLogger(stderr, Logging.Info))



include("../assamble_gpu.jl")
include("../CustomDataStructs/GpuWriteBack.jl")
include("../utils/backend.jl")



const MiB = 2^20
const GiB = 2^30
config = Dict()
config["writeBackStrategy"] = GpuWriteBackFalseInstance()
config["amount_of_gpus"] = 1
config["total_GPU_budget"] = 3 * GiB
config["InstancedoubleQuadRuleGpuStrategyShouldCalculate"] = doubleQuadRuleGpuStrategyShouldCalculateInstance()
config["ShouldCalcInstance"] = ShouldCalcTrueInstance()
config["GPU_budget_pipeline_result"] = 12 * GiB
config["amount_of_producers"] = 16
config["gpu_schedular_print_filename"] = "testing/determin_amount_of_memory_per_thread.txt"

inv_density_factor = 40
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)


samples = 20

filename = "testing/determin_amount_of_memory.txt"

open(filename, "a") do file
    redirect_stdout(file) do
        println(config["GPU_budget_pipeline_result"])
    end
end
for amount_of_memory in [6, 12, 18, 24, 30]
    config["GPU_budget_pipeline_result"] = amount_of_memory  * GiB
    println(amount_of_memory, " ", config["GPU_budget_pipeline_result"])

#     open(config["gpu_schedular_print_filename"], "a") do file
#         println(file, amount_of_memory)
#     end


#     success = false
#     b = 0
#     while !success
#         try
#             b = @benchmark assemble_gpu($S,$X,$X,$config) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
#             success = true
#         catch e
#             open(config["gpu_schedular_print_filename"], "a") do file
#                 println("Error occurred: ", e)
#                 println("Retrying...")
#             end
#         end
#     end
#     b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9

    
#     println(config["GPU_budget_pipeline_result"])
#     println(b_stats)
    
#     open(filename, "a") do file
#         println(file, config["GPU_budget_pipeline_result"])
#         println(file, b_stats)
#     end

end