

using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization

using ImageShow  
# using Images, ImageView
using Plots, Profile, FlameGraphs  


include("BEAST_operator_copy.jl")
include("graph_data.jl")



for inv_density_factor in [30, 33, 36, 38]#density_values
    Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
    X = lagrangec0d1(Γ) 
    S = Helmholtz3D.singlelayer(wavenumber = 1.0)

    assemble_sigle_thread(S,X,X,"data/CPUSingleThread/$(inv_density_factor)", "a")

end

nothing
