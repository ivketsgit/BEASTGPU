using CUDA
using BenchmarkTools

include("../graph_data.jl")

samples = 100
for e in system_matrix_size
    times = []

    A = rand(ComplexF64, e)
    d_array = CuArray(A) 

    
    GC.gc()

    times = @benchmark Array($d_array)  samples=samples evals=1 seconds=3600 * 2

    GC.gc()

    times = times.times / 1e9
    num_runs = length(times)
    total_duration = sum(times)
    open("data/other/transfer_warm_run_server/$(e).txt", "a") do file
        println(file, """
                Manual Benchmark of GPU-to-CPU transfer duration $(total_duration) seconds over $(num_runs) runs:
                $(times)
                """)
    end
end
