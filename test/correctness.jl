using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization

include("../assamble_gpu.jl")
include("../CustomDataStructs/GpuWriteBack.jl")
include("../utils/configuration.jl")
include("configers.jl")
include("difference.jl")

function silence(f::Function, args...; kwargs...)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return f(args...; kwargs...)
        end
    end
end


inv_density_factor = 1
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
filename = "cashed_results/matrix_ref_$inv_density_factor.bin"

let 
    M_ref = open(filename, "r") do io
        deserialize(io)
    end

    configs = generate_configs()

    for config in configs
        # print_fields(config)
        M = silence(assemble_gpu,S,X,X,config,config.writeBackStrategy)
        # print("\n")
        # @show test(M, M_ref)
        @assert(test(M, M_ref) < 10^-10, "The result of the GPU assembly does not match the reference assembly.")
        # print("\n\n\n\n\n\n")
    end
    print("All tests passed successfully!\n")
end




