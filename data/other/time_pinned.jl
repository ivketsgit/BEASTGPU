using CUDA
using BenchmarkTools

samples = 100
for e in range(1, 10)
    times = []
    for i in 1:samples
        GC.gc()
        t = @elapsed begin
            CUDA.pin(Array{ComplexF64}(undef, Int(round(38400*(e/10))),38400))
        end
        if i % 10 == 0
            print(".")
        end
    end 
    push!(times, t)

    GC.gc()


    num_runs = length(times)
    total_duration = sum(times)
    open("data/other/pinned_server/$(e).txt", "a") do file
        println(file, """
                Manual Benchmark of duration $(total_duration) seconds over $(num_runs) runs:"
                $(times)
                """
        )
    end


end
