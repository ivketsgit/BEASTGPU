using CUDA
using BenchmarkTools

include("../graph_data.jl")

samples = 100
for e in system_matrix_size
    times = []

    A = rand(ComplexF64, e)
    d_array = CuArray(A) 
    CPU_array = A

    for i in 1:samples
        GC.gc()
        t = @elapsed begin
            copyto!(CPU_array, d_array)
        end
        push!(times, t)
        if i % 10 == 0
            print(".")
        end
    end

    GC.gc()

    num_runs = length(times)
    total_duration = sum(times)
    open("data/other/transfer_server/$(e).txt", "a") do file
        println(file, """
                Manual Benchmark of GPU-to-CPU transfer duration $(total_duration) seconds over $(num_runs) runs:
                $(times)
                """)
    end
end
