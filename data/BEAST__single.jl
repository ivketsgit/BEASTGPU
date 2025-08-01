using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization

include("graph_data.jl")

samples = 100
for inv_density_factor in density_values
    Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
    X = lagrangec0d1(Γ) 
    S = Helmholtz3D.singlelayer(wavenumber = 1.0)


    times = @benchmark BEAST.assemble($S,$X,$X,threading=$BEAST.Threading{:single}) samples=samples evals=1 seconds=3600 * 2
    
    
    num_runs = length(times.times)
    total_duration = sum(times.times) / 1e9
    open("data/full_single/$(inv_density_factor)/full_time.txt", "a") do file
        println(file, """
            Manual Benchmark of duration $(total_duration) seconds over $(num_runs) runs:"
            $(times.times / 1e9)
            """
        )
    end
end