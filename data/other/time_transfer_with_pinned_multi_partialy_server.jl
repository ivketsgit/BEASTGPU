using CUDA
using BenchmarkTools
using KernelAbstractions

include("../graph_data.jl")
include("../../utils/copy_to_CPU.jl")



CUDA.allowscalar(false)
backend = CUDABackend()

samples = 100
for e in system_matrix_size
    times = []

    A = rand(ComplexF64, (Int(ceil(sqrt(e))), Int(ceil(sqrt(e)))))
    d_array = CuArray(A) 
    
    CPU_array = CUDA.pin(Array{ComplexF64}(undef,    (Int(ceil(sqrt(e))), Int(ceil(sqrt(e))))))
    time_subparts = [[], []]

    GC.gc()

    

    result_cpu = Array{ComplexF64}(undef,   (Int(ceil(sqrt(e))), Int(ceil(sqrt(e)))))
    complex_array = KernelAbstractions.allocate(backend, ComplexF64, size(result_cpu))
    chunk_size = Int(ceil(prod(size(result_cpu))/100))

    # Benchmark
    times = @benchmark begin
        copy_to_CPU($result_cpu, $complex_array, $backend, $ComplexF64, $chunk_size, $nothing,  $time_subparts)
    end samples=samples evals=1 seconds=3600 * 2



    # times = @benchmark copyto!($CPU_array, $d_array)  samples=samples evals=1 seconds=3600 * 2

    GC.gc()

    times = times.times / 1e9 
    num_runs = length(times)
    total_duration = sum(times) 
    open("data/other/transfer_with_pinned_multi_partialy_server/$(e)_full.txt", "a") do file
        println(file, """
                Manual Benchmark of GPU-to-CPU transfer duration $(total_duration) seconds over $(num_runs) runs:
                $(times)
                """)
    end
    open("data/other/transfer_with_pinned_multi_partialy_server/$(e)_allocation.txt", "a") do file
        println(file, """
                Manual Benchmark of GPU-to-CPU transfer duration $(total_duration) seconds over $(num_runs) runs:
                $(time_subparts[1])
                """)
    end
    open("data/other/transfer_with_pinned_multi_partialy_server/$(e)_transfer.txt", "a") do file
        println(file, """
                Manual Benchmark of GPU-to-CPU transfer duration $(total_duration) seconds over $(num_runs) runs:
                $(time_subparts[2])
                """)
    end
end