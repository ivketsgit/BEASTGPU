using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization
using Logging
using LoggingExtras

# global_logger(ConsoleLogger(stderr, Logging.Info))

# file_logger = FileLogger("testing/amdahl's_law_2_results_log.txt", Logging.Info)
# global_logger(file_logger)

# io = open("testing/amdahl's_law_2_results_log.txt", "a")
# file_logger = FileLogger(io, Logging.Info)
# global_logger(file_logger)

# file_logger = LoggingExtras.FileLogger("testing/amdahl's_law_2_results_log.txt"; append=true, level=Logging.Info)
# global_logger(file_logger)

# import Base.Filesystem: mkpath
# mkpath("testing")

# # Open the file stream
# log_stream = open("testing/amdahl's_law_2_results_log.txt", "a")

# # Create a logger that writes to the stream
# stream_logger = StreamLogger(log_stream)

# # Filter to Info level and above
# filtered_logger = MinLevelLogger(stream_logger, Logging.Info)

# # Set as global logger
# global_logger(filtered_logger)

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
writeBackStrategy = GpuWriteBackTrueInstance()
configuration = Dict()
configuration["writeBackStrategy"] = writeBackStrategy
configuration["amount_of_gpus"] = 1
configuration["total_GPU_budget"] = 3 * GiB
configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"] = doubleQuadRuleGpuStrategyShouldCalculateInstance()
configuration["ShouldCalcInstance"] = ShouldCalcTrueInstance()
configuration["GPU_budget_pipeline_result"] = 12 * GiB
configuration["amount_of_producers"] = 16



# global time_logger = Dict()
# time_logger["calculate the double int"] = []
# time_logger["calculate the double int"] = []
# time_logger["transfer quadrule to CPU"] = []
# time_logger["calculate double for loop"] = []
# time_logger["calculate SauterSchwab"] = []
# time_logger["time_table[1,:]"] = []
# time_logger["time_table[2,:]"] = []
# time_logger["time_to_store"] = []
# time_logger["transfer results to CPU"] = []
# time_logger["create results as complex numbers"] = []
# time_logger["time_sauter_schwab_overhead_and_test_toll 2"] = []
# time_logger["time_sauter_schwab_overhead_and_test_toll 3"] = []
# time_logger["time_sauter_schwab_overhead_and_test_toll 4"] = []


# global function log_time(time_logger, key::string, value::Float64)
#     if haskey(time_logger, key)
#         push!(time_logger[key], value)
#     else
#         time_logger[key] = [value]
#     end
# end


inv_density_factor = 1
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
BEAST.assemble(S,X,X)
assemble_gpu(S,X,X,configuration)




# for e in time_logger
#     println(e
# end
# throw("test")


inv_density_factor = 40
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
filename = "testing/amdahl's_law_2_results.txt"




open(filename, "a") do file
    redirect_stdout(file) do
        println("start")
    end
end

# a = @benchmark BEAST.assemble($S,$X,$X)
# display(a)



samples = 20





# configuration["amount_of_gpus"] = 1
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     redirect_stdout(file) do
#         println(configuration["amount_of_gpus"]
#         println(b_stats
#     end
# end
# println(configuration["amount_of_gpus"]
# println(b_stats



# configuration["amount_of_gpus"] = 5
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     redirect_stdout(file) do
#         println(configuration["amount_of_gpus"])
#         println(b_stats)
#     end
# end
# println(configuration["amount_of_gpus"]
# println(b_stats

     

# configuration["amount_of_gpus"] = 16
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     redirect_stdout(file) do
#         println(configuration["amount_of_gpus"])
#         println(b_stats)
#     end
# end
# println(configuration["amount_of_gpus"]
# println(b_stats


# configuration["amount_of_gpus"] = 64
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     redirect_stdout(file) do
#         println(configuration["amount_of_gpus"])
#         println(b_stats)
#     end
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)

# configuration["amount_of_gpus"] = 256
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     redirect_stdout(file) do
#         println(configuration["amount_of_gpus"])
#         println(b_stats)
#     end
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)



# configuration["amount_of_gpus"] = 1024
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     println(file, configuration["amount_of_gpus"])
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)

writeBackStrategy = GpuWriteBackFalseInstance()
configuration["writeBackStrategy"] = writeBackStrategy


configuration["amount_of_gpus"] = 1
open("testing/a_dump.txt", "a") do file
    println(file, "\n", configuration["amount_of_gpus"], "\n")
end
b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
open(filename, "a") do file
    println(file, configuration["amount_of_gpus"])
    println(file, b_stats)
end
println(configuration["amount_of_gpus"])
println(b_stats)


configuration["amount_of_gpus"] = 5
open("testing/a_dump.txt", "a") do file
    println(file, "\n", configuration["amount_of_gpus"], "\n")
end
b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
open(filename, "a") do file
    println(file, configuration["amount_of_gpus"])
    println(file, b_stats)
end
println(configuration["amount_of_gpus"])
println(b_stats)


configuration["amount_of_gpus"] = 16
open("testing/a_dump.txt", "a") do file
    println(file, "\n", configuration["amount_of_gpus"], "\n")
end
b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
open(filename, "a") do file
    println(file, configuration["amount_of_gpus"])
    println(file, b_stats)
end
println(configuration["amount_of_gpus"])
println(b_stats)


# configuration["amount_of_gpus"] = 2
# open("testing/a_dump.txt", "a") do file
#     println(file, "\n", configuration["amount_of_gpus"], "\n")
# end
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     println(file, configuration["amount_of_gpus"])
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)







# configuration["amount_of_gpus"] = 4
# open("testing/a_dump.txt", "a") do file
#     println(file, "\n", configuration["amount_of_gpus"], "\n")
# end
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     println(file, configuration["amount_of_gpus"])
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)

 








# configuration["amount_of_gpus"] = 8
# open("testing/a_dump.txt", "a") do file
#     println(file, "\n", configuration["amount_of_gpus"], "\n")
# end    
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     println(file, configuration["amount_of_gpus"])
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)



# configuration["amount_of_gpus"] = 10
# open("testing/a_dump.txt", "a") do file
#     println(file, "\n", configuration["amount_of_gpus"], "\n")
# end
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     println(file, configuration["amount_of_gpus"])
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)


# configuration["amount_of_gpus"] = 12
# open("testing/a_dump.txt", "a") do file
#     println(file, "\n", configuration["amount_of_gpus"], "\n")
# end
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     println(file, configuration["amount_of_gpus"])
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)



# configuration["amount_of_gpus"] = 14
# open("testing/a_dump.txt", "a") do file
#     println(file, "\n", configuration["amount_of_gpus"], "\n")
# end
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     println(file, configuration["amount_of_gpus"])
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)



# configuration["amount_of_gpus"] = 16
# open("testing/a_dump.txt", "a") do file
#     println(file, "\n", configuration["amount_of_gpus"], "\n")
# end
# b = @benchmark assemble_gpu($S,$X,$X,$configuration) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
# b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
# open(filename, "a") do file
#     println(file, configuration["amount_of_gpus"])
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)



# # open("testing/a_dump.txt", "a") do file
# #     println(file, "\n", configuration["amount_of_gpus"], "\n")
# # end
# # println(configuration["amount_of_gpus"])
# # println(b_stats)






# open(filename, "a") do file
#     println(file,"BEAST.assemble")
#     b = @benchmark BEAST.assemble(S,X,X) samples=samples evals=1 seconds=99999999999999999999999999999999999999999999999999
#     b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
#     println(file, b_stats)
# end
# println(configuration["amount_of_gpus"])
# println(b_stats)