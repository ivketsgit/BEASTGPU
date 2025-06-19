using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization
using Logging
using LoggingExtras


struct FileInfoLogger <: AbstractLogger
    io::IO
end

function Logging.shouldlog(logger::FileInfoLogger, level, _module, group, id)
    return level >= Info  # Only log Info and above
end

function Logging.min_enabled_level(::FileInfoLogger)
    return Info
end

function Logging.catch_exceptions(::FileInfoLogger)
    return false
end

function Logging.handle_message(logger::FileInfoLogger, level, message, _module, group, id, file, line; kwargs...)
    println(logger.io, "[$(level)] $(message)")
    flush(logger.io)
end

# Make sure the directory exists
mkpath("testing")
io = open("testing/amdahl's_law_2_results_log.txt", "a")

# Set global logger
global_logger(FileInfoLogger(io))


include("../assamble_gpu.jl")
include("../CustomDataStructs/GpuWriteBack.jl")
include("../utils/backend.jl")


# const B = 8
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
# config["gpu_schedular_print_filename"] = "testing/a_dump_3.txt"

global time_logger = Dict()
time_logger["calculate the double int"] = []
time_logger["calculate the double int"] = []
time_logger["transfer quadrule to CPU"] = []
time_logger["calculate double for loop"] = []
time_logger["calculate SauterSchwab"] = []
time_logger["time_table[1,:]"] = []
time_logger["time_table[2,:]"] = []
time_logger["time_to_store"] = []
time_logger["transfer results to CPU"] = []
time_logger["create results as complex numbers"] = []
time_logger["time_sauter_schwab_overhead_and_test_toll 2"] = []
time_logger["time_sauter_schwab_overhead_and_test_toll 3"] = []
time_logger["time_sauter_schwab_overhead_and_test_toll 4"] = []


#warmup
inv_density_factor = 1
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
BEAST.assemble(S,X,X)
assemble_gpu(S,X,X,config)

inv_density_factor = 40
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
filename = "testing/amdahl's_law_3_results_2.txt"


open(filename, "a") do file
    redirect_stdout(file) do
        println("start")
    end
end

samples = 20

# config["writeBackStrategy"] = GpuWriteBackFalseInstance()
# open(filename, "a") do file
#     redirect_stdout(file) do
#         println(config["writeBackStrategy"])
#     end
# end
# for amount_of_gpus in [1, 2, 3, 4, 6, 12, 102] # 
#     config["amount_of_gpus"] = amount_of_gpus

#     # open(config["gpu_schedular_print_filename"], "a") do file
#     #     println(file, config["amount_of_gpus"])
#     # end

#     # if amount_of_gpus == 1
#     #     open("testing/time_for_supartts_.txt", "a") do file
#     #         println(file, config["amount_of_gpus"])
#     #     end
#     # end

#     println(amount_of_gpus)
#     assemble_gpu(S,X,X,config) 
#     # b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9

    
#     # println(config["amount_of_gpus"])
#     # println(b_stats)

#     # open(filename, "a") do file
#     #     println(file, config["amount_of_gpus"])
#     #     println(file, b_stats)
#     #     end
#     # end

# end







config["writeBackStrategy"] = GpuWriteBackTrueInstance()
open(filename, "a") do file
    redirect_stdout(file) do
        println(config["writeBackStrategy"])
    end
end
for amount_of_gpus in [1, 2, 3, 4, 6, 12, 102]
    config["amount_of_gpus"] = amount_of_gpus
    b = @benchmark assemble_gpu($S,$X,$X,$config) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
    b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
    open(filename, "a") do file
        redirect_stdout(file) do
            println(config["amount_of_gpus"])
            println(b_stats)
        end
    end
    println(config["amount_of_gpus"])
    println(b_stats)
end











# open(filename, "a") do file
#     println(file,"BEAST.assemble")
#     b = @benchmark BEAST.assemble(S,X,X) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
#     b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
#     println(file, b_stats)
# end
# println(config["amount_of_gpus"])
# println(b_stats)