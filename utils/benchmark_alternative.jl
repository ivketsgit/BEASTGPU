using Dates
using Statistics

const PRINT_LOCK = ReentrantLock()
const _warmed_up_functions = Set{UInt64}()
function manual_benchmark(f; args=(), kwargs=NamedTuple(), n=1000, max_hours=1,filename=nothing,appendOrWrite="a")
     # Hash the function to identify it uniquely
    f_hash = hash(f)

    # Only warm up if this function hasn't been warmed up before
    if !(f_hash in _warmed_up_functions)
        f(args...; kwargs...)  # Warm-up to trigger compilation
        push!(_warmed_up_functions, f_hash)
    end

    # Avoid global lookup by creating closure
    benchmark_closure = () -> f(args...; kwargs...)

    times = Float64[]
    t0 = time()  # Wall-clock time in seconds
    i = 1
    break_time = 60 * 60 * max_hours
    for j in 1:n
        if (time() - t0) > break_time && j > 5
            println("Breaking early: Benchmark exceeded $(break_time/60) minute.")
            break
        end
        attempt = 0
        while true
            try
                GC.gc()
                CUDA.reclaim()
                t_start = time_ns()
                benchmark_closure()
                t_end = time_ns()
                GC.gc()
                CUDA.reclaim()
                push!(times, (t_end - t_start) / 1e9)  # convert to s
                i = j
                break
            catch e
                attempt += 1
                if isa(e, CUDA.CuError) && e.code == CUDA.cudaErrorHostMemoryAlreadyRegistered
                    GC.gc()
                    CUDA.reclaim()
                elseif isa(e, TaskFailedException)
                    GC.gc()
                    CUDA.reclaim()
                else
                    @error "Unhandled error in benchmark run" exception=(e, catch_backtrace())
                end

                if attempt == 50
                    # lock(PRINT_LOCK) do
                    #     open(filename,  appendOrWrite) do file
                    #         # println(file, """
                    #         # Manual Benchmark of duration $((time() - t0)) over $i runs:
                    #         # Min: $(minimum(times)) s
                    #         # Mean: $(mean(times)) s
                    #         # Max: $(maximum(times)) s
                    #         # Std: $(std(times)) s
                    #         # 2nd Quartile (Median): $(quantile(times, 0.5)) s
                    #         # 3rd Quartile (75th percentile): $(quantile(times, 0.75)) s
                    #         # """)

                    #         println(file, """
                    #         Manual Benchmark of duration $((time() - t0)) over $i runs:
                    #         $(times)
                    #         """)
                    #     end
                    # end
                    error("Benchmark failed after 50 attempts.")
                end
            end
        end
    end

    # lock(PRINT_LOCK) do
        
    #     open(filename,  appendOrWrite) do file
    #         # println(file, """
    #         # Manual Benchmark of duration $((time() - t0)) over $i runs:
    #         # Min: $(minimum(times)) s
    #         # Mean: $(mean(times)) s
    #         # Max: $(maximum(times)) s
    #         # Std: $(std(times)) s
    #         # 2nd Quartile (Median): $(quantile(times, 0.5)) s
    #         # 3rd Quartile (75th percentile): $(quantile(times, 0.75)) s
    #         # """)

    #         println(file, """
    #         Manual Benchmark of duration $((time() - t0)) over $i runs:
    #         $(times)
    #         """)
    #     end
        
    # end

    
    # println("Manual Benchmark of duration $((time() - t0)) over $i runs:")
    # println("  Min: $(minimum(times)) s")
    # println("  Mean: $(mean(times)) s")
    # println("  Max: $(maximum(times)) s")
    # println("  Std: $(std(times)) s")
    # println("  2nd Quartile (Median): $(quantile(times, 0.5)) s")
    # println("  3rd Quartile (75th percentile): $(quantile(times, 0.75)) s")

    return times
end

