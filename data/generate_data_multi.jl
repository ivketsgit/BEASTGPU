

using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization
using CUDA

using ImageShow  
# using Images, ImageView
using Plots, Profile, FlameGraphs  


include("BEAST_operator_copy.jl")
include("graph_data.jl")



for inv_density_factor in density_values
    Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
    X = lagrangec0d1(Γ) 
    S = Helmholtz3D.singlelayer(wavenumber = 1.0)

    assemble_multi_thread(S,X,X,"data/CPUMultiThread/$(inv_density_factor)", "a")

end

nothing
