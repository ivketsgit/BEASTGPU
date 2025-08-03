using CUDA
using BenchmarkTools


samples = 100
for e in [1474713604]
    times = []

    A = rand(ComplexF64, e)
    d_array = CuArray(A) 

    # GC.gc()

    for _ in 1:samples
        t = @elapsed begin
            h_array = Array(d_array)
        end
        @show 1474713604*16 / t / 1e9
    end


    # times = @benchmark Array($d_array)  samples=samples evals=1 seconds=3600 * 2

    # GC.gc()
    # println(times.times)

    # times = times.times / 1e9 
    # num_runs = length(times)
    # total_duration = sum(times) 
    # open("data/other/transfer_warm_run_home_pc/$(e).txt", "a") do file
    #     println(file, """
    #             Manual Benchmark of GPU-to-CPU transfer duration $(total_duration) seconds over $(num_runs) runs:
    #             $(times)
    #             """)
    # end
end
