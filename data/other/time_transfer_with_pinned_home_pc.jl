using CUDA
using BenchmarkTools

include("../graph_data.jl")


function threaded_copy!(CPU_array, d_array, chunk_size)
    Threads.@threads for i in 1:Threads.nthreads()
        start_idx = (i - 1) * chunk_size + 1
        end_idx = min(i * chunk_size, length(CPU_array))
        if start_idx <= end_idx
            copyto!(CPU_array, start_idx, d_array, start_idx, end_idx - start_idx + 1)
        end
    end
    return nothing
end


samples = 100
for e in system_matrix_size
    times = []

    A = rand(ComplexF64, e)
    d_array = CuArray(A) 
    
    CPU_array = CUDA.pin(Array{ComplexF64}(undef,  e))

    GC.gc()

    nthreads = Threads.nthreads()
    chunk_size = cld(e, nthreads)



    # Benchmark
    times = @benchmark begin
        threaded_copy!($CPU_array, $d_array, $chunk_size)
    end samples=samples evals=1 seconds=3600 * 2


    # times = @benchmark copyto!($CPU_array, $d_array)  samples=samples evals=1 seconds=3600 * 2

    GC.gc()

    times = times.times / 1e9 
    num_runs = length(times)
    total_duration = sum(times) 
    open("data/other/transfer_with_pinned_home_pc/$(e).txt", "a") do file
        println(file, """
                Manual Benchmark of GPU-to-CPU transfer duration $(total_duration) seconds over $(num_runs) runs:
                $(times)
                """)
    end
end
