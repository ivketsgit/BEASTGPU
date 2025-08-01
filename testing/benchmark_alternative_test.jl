include("../utils/benchmark_alternative.jl")

using BenchmarkTools
using Statistics




# === Test function and comparison ===
function test_comparison(x, y, n, bt_result)
    println("\nRunning benchmark comparison test...")

    # Run our manual benchmark
    manual_times = manual_benchmark(test_func; args=(x, y), n=n)

    # Compare metrics
    bt_min = minimum(bt_result).time / 1e6  # ns to ms
    mb_min = minimum(manual_times)

    println("\nComparison:")
    println("  BenchmarkTools min: $(bt_min) ms")
    println("  ManualBenchmark min: $(mb_min) ms")

    # Optional: assert close (not strict)
    if abs(bt_min - mb_min) / bt_min < 0.5
        println("✅ Manual benchmark is in the expected range.")
    else
        println("⚠️ Significant discrepancy detected.")
    end
end



# Example function to benchmark
function test_func(x, y)
    result = x * y + sin(x)
    # Artificial computation to consume time
    acc = 0.0
    for i in 1:10^7
        acc += sin(i) * cos(i)
    end
    return result + acc * 1e-16  # scale down so it doesn't affect result much
end
# test_func(x, y) = x * y + sin(x)

x = 3.14
y = 2.71
n = 1000

println("\nBenchmarkTools result:")
@benchmark test_func($x, $y)
@btime test_func($x, $y)
b = @benchmark test_func($x, $y)

b_stats = BenchmarkTools.mean(b).time / 10^9, BenchmarkTools.std(b).time / 10^9, minimum(b).time / 10^9, maximum(b).time / 10^9
    
println(b_stats)

println(b)


# Run test
test_comparison(x, y, n, b)
