using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization
using Logging

#global_logger(ConsoleLogger(stderr, Logging.Info))


include("assamble_gpu.jl")
include("CustomDataStructs/GpuWriteBack.jl")
include("utils/backend.jl")



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
configuration["writeBackStrategy"] = GpuWriteBackTrueInstance()
configuration["amount_of_gpus"] = 1
configuration["total_GPU_budget"] = 3 * GiB
configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"] = doubleQuadRuleGpuStrategyShouldCalculateInstance()
configuration["ShouldCalcInstance"] = ShouldCalcTrueInstance()
configuration["GPU_budget_pipeline_result"] = 24 * GiB
configuration["amount_of_producers"] = 16

inv_density_factor = 40
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
# Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor; generator=:gmsh)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
filename = "cashed_results/matrix_ref_$inv_density_factor.bin"


# @show dimension(Γ)
# vertices = skeleton(Γ, 0)
# num_nodes = length(vertices)
# @show num_nodes

# let time = @elapsed begin
#    # @show @which assemble(S,X,X)
#        M_ref = BEAST.assemble(S,X,X)
#    end
#    open(filename, "w") do io
#        serialize(io, M_ref)
#    end
#    println("Elapsed time control: ", time)
#    println("")
# end


global time_logger = Dict()
time_logger["time overhead"] = []
time_logger["time to determin the quadrule"] = []
time_logger["calculate the double int"] = []
time_logger["transfer quadrule to CPU"] = []
time_logger["calculate double for loop"] = []
time_logger["calculate SauterSchwab"] = []
# time_logger["time_table[1,:]"] = []
# time_logger["time_table[2,:]"] = []
time_logger["time_to_store"] = []
time_logger["transfer results to CPU"] = []
time_logger["create results as complex numbers"] = []

time_logger["time_sauter_schwab_overhead_and_test_toll 2"] = []
time_logger["time_sauter_schwab_overhead_and_test_toll 3"] = []
time_logger["time_sauter_schwab_overhead_and_test_toll 4"] = []

time_logger["calc_sauter_schwab 2"] = []
time_logger["calc_sauter_schwab 3"] = []
time_logger["calc_sauter_schwab 4"] = []

let time = @elapsed begin
        M_ref_gpu = assemble_gpu(S,X,X,configuration)
    end 
    println("Elapsed time: ", time)
    println("")
end


# function extract_atomic_values(value)
#     @show value
#     if value == []
#         # Case 3: Empty vector
#         return []
#     elseif all(x -> isa(x, Atomic{Float64}), value)
#         # Case 1: Flat vector of Atomic{Float64}
#         @show value
#         return mean([x[] for x in value][:])
#     elseif all(x -> isa(x, Vector{Atomic{Float64}}), value)
#         # Case 2: Vector of Vectors (2D matrix-like structure)
#         return [mean([x[] for x in row][:]) for row in value]
#     elseif all(x -> isa(x, Float64), value)
#         # Case 3: Empty vector
#         return mean(value[:])
#     elseif all(x -> isa(x, Vector{Float64}), value)
#         # Case 3: Empty vector
#         return mean(value[:])
#     else
#         @show value
#         @show typeof(value)
#         error("Unsupported structure: expected Vector{Atomic{Float64}} or Vector{Vector{Atomic{Float64}}}")
#     end
# end

# let keys = ["time overhead", "time to determin the quadrule", "calculate the double int", "transfer quadrule to CPU", "calculate double for loop", "calculate SauterSchwab", "time_to_store", "transfer results to CPU", "create results as complex numbers", "time_sauter_schwab_overhead_and_test_toll 2", "time_sauter_schwab_overhead_and_test_toll 3", "time_sauter_schwab_overhead_and_test_toll 4"]
#     for key in keys
#         if haskey(time_logger, key)
#             value = time_logger[key]
#             means = extract_atomic_values(value)

#             println(key, "     ", means)
#             # @show value
#         end
#     end
# end

# let time = @elapsed begin
#         M = assemble_gpu(S,X,X,writeBackStrategy,2)
#     end 
#     println("Elapsed time: ", time)
#     println("")
# end
# @show M_ref
# @show M

# M_ref = open(filename, "r") do io
#     deserialize(io)
# end

# # println("")

# global_logger(ConsoleLogger(stderr, Logging.Info)) 
# # error_matrix = abs.(M_ref .- M)
# # @show maximum(error_matrix)
# error_matrix = abs.(M_ref .- M_ref_gpu)
# println("")
# @show maximum(error_matrix)

 
# error_matrix = abs.(M_ref_gpu .- M)
# @show maximum(error_matrix)
# @show M
# @show M_ref
# for (i, e) in enumerate(abs.(M_ref .- M))
#     @show i, e
# end
