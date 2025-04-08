using KernelAbstractions


include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend



@kernel function tensor(results, @Const(V1), @Const(V2))
    i, j = @index(Global, NTuple)
    results[i, j] = V1[i] + V2[j]
end


ch = Channel{Tuple{Array{ComplexF64, 3}, Tuple{Int64, Int64}, Tuple{Int64, Int64}}}(blocks*blocks)  
t = Threads.@spawn consumer(ch, store, length_return_matrix, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, time_to_store)


function consumer(ch::Channel, store, length_return_matrix, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, time_to_store)
    for item in ch
        time_to_store += @elapsed begin
            result_cpu, prev_range, prev_offsets = item
            write_to_compact_matrix(result_cpu, store, length_return_matrix, prev_range, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, prev_offsets)
        end
    end
end