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
configuration = Dict()
configuration["writeBackStrategy"] = GpuWriteBackFalseInstance()
configuration["amount_of_gpus"] = 1
configuration["total_GPU_budget"] = 3 * GiB
configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"] = doubleQuadRuleGpuStrategyShouldCalculateInstance()
configuration["ShouldCalcInstance"] = ShouldCalcTrueInstance()
configuration["GPU_budget_pipeline_result"] = 12 * GiB
configuration["amount_of_producers"] = 16
configuration["gpu_schedular_print_filename"] = "testing/determin_amount_of_threads_per_thread.txt"

inv_density_factor = 40
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)


samples = 20

filename = "testing/determin_amount_of_threads.txt"

open(filename, "a") do file
    redirect_stdout(file) do
        println(configuration["amount_of_producers"])
    end
end
for amount_of_producers in [2,4,8,10,11,12,13,14,16]
    configuration["amount_of_producers"] = amount_of_producers

    open(configuration["gpu_schedular_print_filename"], "a") do file
        println(file, configuration["amount_of_producers"])
    end

    success = false
    b = 0
    while !success
        try
            b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
            success = true
        catch e
            open(configuration["gpu_schedular_print_filename"], "a") do file
                println("Error occurred: ", e)
                println("Retrying...")
            end
        end
    end
    b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9

    
    println(configuration["amount_of_producers"])
    println(b_stats)
    
    open(filename, "a") do file
        println(file, configuration["amount_of_producers"])
        println(file, b_stats)
    end

end