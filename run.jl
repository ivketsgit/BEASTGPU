using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization

using ImageShow  
# using Images, ImageView
using Plots, Profile, FlameGraphs    


include("assamble_gpu.jl")
include("CustomDataStructs/GpuWriteBack.jl")
include("utils/configuration.jl")



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
CUDA.allowscalar(false)
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
        false,
        ""
    )


inv_density_factor = 24
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
# Profile.init(n = 10_000_000) 
# Profile.clear(); 
# assemble_gpu(S,X,X,config,config.writeBackStrategy)

let time = @elapsed begin
        M = assemble_gpu(S,X,X,config,config.writeBackStrategy)

        
# using Profile
#                     Profile.init(delay = 0.001)  # 1ms between samples

#                     Profile.clear()              # Clear any old data
#                     @profile begin
#                         assemble_gpu(S,X,X,config,config.writeBackStrategy)
#                     end

#                     Profile.print()              # Or use ProfileView.jl for a GUI

#                     Profile.print(format=:flat)

        # 
        # @profview assemble_gpu(S,X,X,config,config.writeBackStrategy)

        
        
        # @profview assemble_gpu(S,X,X,config,config.writeBackStrategy)

        # @profview_allocs assemble_gpu(S,X,X,config,config.writeBackStrategy) sample_rate = 1
        
        # @profile for _ in 1:1000
        #     assemble_gpu(S, X, X, config, config.writeBackStrategy)
        # end
    end 
    println("Elapsed time: ", time)
    println("")
end
GC.gc()
# Profile.print()     
# g = flamegraph()
# g = flamegraph(C=true)
# img = flamepixels(g)
# save("flamegraph.png", img)
# display(img)
# imshow(img)






# print_means(config.timeLogger)



# M_ref = open(filename, "r") do io
#     deserialize(io)
# end

# min_M_row = Array{Float64}(undef, size(M)[1])
# @threads for col in 1:size(M)[1]
#     min_M_row[col] = abs.(M_ref[col] .- M[col])
# end
# @show maximum(min_M_row)
