using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization

include("../assamble_gpu.jl")
include("../utils/backend.jl")

writeBackStrategy = GpuWriteBackTrueInstance()

inv_density_factor = 40
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)


@btime BEAST.assemble(S,X,X)

# assemble_gpu(S, X, X, writeBackStrategy,1)

# assemble_gpu(S, X, X, writeBackStrategy,4)

# assemble_gpu(S, X, X, writeBackStrategy,16)

# assemble_gpu(S, X, X, writeBackStrategy,64)

# assemble_gpu(S, X, X, writeBackStrategy,256)

# assemble_gpu(S, X, X, writeBackStrategy,1024)




println("")

# error_matrix = abs.(M_ref .- M)
# @show maximum(error_matrix)
# error_matrix = abs.(M_ref .- M_ref_gpu)
# @show maximum(error_matrix)