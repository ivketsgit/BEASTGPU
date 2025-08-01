using CUDA
using BenchmarkTools

include("../graph_data.jl")

samples = 100
for e in system_matrix_size
    times = []
    for i in 1:samples
        GC.gc()
        t = @elapsed begin
            CUDA.pin(Array{ComplexF64}(undef, e))
        end
        push!(times, t)
        if i % 10 == 0
            print(".")
        end
    end 

    GC.gc()


    num_runs = length(times)
    total_duration = sum(times)
    open("data/other/pinned_home_pc/$(e).txt", "a") do file
        println(file, """
                Manual Benchmark of duration $(total_duration) seconds over $(num_runs) runs:"
                $(times)
                """
        )
    end


end
