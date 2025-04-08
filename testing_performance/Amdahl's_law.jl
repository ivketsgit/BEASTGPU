using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization

include("../assamble_gpu.jl")
include("../utils/backend.jl")



inv_density_factor = 1
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
filename = "zzz/cashed_results/matrix_ref_$inv_density_factor.bin"

let time = @elapsed begin
    # @show @which assemble(S,X,X)
        M_ref = assemble(S,X,X)
    end
    open(filename, "w") do io
        serialize(io, M_ref)
    end
    println("Elapsed time control: ", time)
    println("")
end

let time = @elapsed begin
        M = assemble_gpu(S,X,X)
    end 
    println("Elapsed time: ", time)
    println("")
end