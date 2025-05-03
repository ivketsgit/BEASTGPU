include("utils/noSwitch.jl")

function producer(ch_data::Channel, backend,
        assembly_gpu_data,
        biop,
        wimps_and_womps, 
        time_table, should_calc, size_submatrix,

        store, length_return_matrix, blocks,
        elements_data, floatType,
        configuration
    )
    result_cpu = Array{ComplexF64}(undef, size_submatrix, size_submatrix, 9)
    results = KernelAbstractions.allocate(backend, ComplexF64, size_submatrix, size_submatrix, 9)
    t1 = 0
    t2 = 0
    t3 = 0
    GC.@preserve results begin
        for item in ch_data
            t1 += @elapsed begin
            i, j, ndrange, curr_offsets = item
            doubleQuadRule_generic_3d_gpu_outside_loop!(results,
                                assembly_gpu_data,
                                biop,
                                wimps_and_womps, 
                                (i - 1) * size_submatrix, (j - 1) * size_submatrix,
                                time_table, 1, ndrange, 
                                elements_data, floatType,
                                configuration,
                                should_calc
                            )

                            

            # yield()
            # result_cpu = Array(results)
            end
            t2 += @elapsed begin
                # @noswitch begin
                    KernelAbstractions.copyto!(CPU(), result_cpu, results)
                # end
                # KernelAbstractions.synchronize(CPU())
            
            end
            t3 += @elapsed begin
            # result_cpu_temp = Array(results)
            
            # result_cpu = Array(results)
            # write_to_compact_matrix(result_cpu, store, length_return_matrix, ndrange, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, curr_offsets)
            linear_index = (j - 1) * blocks + i
            # if !(j == 1 && i == 1)
            #     while finished[linear_index - 1] == 0
            #         yield()
            #     end
            # end
            # @lock lock write_to_compact_matrix(result_cpu, store, length_return_matrix, ndrange, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, curr_offsets)
            if typeof(configuration["writeBackStrategy"]) == GpuWriteBackFalseInstance
                test_assembly_cpu_indexes = assembly_gpu_data[5]
                trial_assembly_cpu_indexes = assembly_gpu_data[6]
                write_to_compact_matrix(result_cpu, store, length_return_matrix, ndrange, configuration["writeBackStrategy"], configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"], test_assembly_cpu_indexes, trial_assembly_cpu_indexes, curr_offsets)
            end
            end
            # item = (result_cpu, ndrange, curr_offsets, i, j)
            # put!(ch_store, item)
            # println("produced ",i, " ",j)
        end
    end
    print("[",t1, ", ", t2, ", ", t3, "],")
end

function load_data(backend, type, elements_length, qd_points, pref_offet, length_1, length_2 = 3)
    weights = Array{type}(undef, elements_length, length_1)
    values = Array{type}(undef, elements_length, length_1, length_2)
    cart =   Array{type}(undef, elements_length, length_1, length_2)

    for i in 1:elements_length
        qd_points_1i = qd_points[1,i + pref_offet]
        @assert size(qd_points_1i)[1] == length_1
        for j in 1:length_1
            weights[i,j] = qd_points_1i[j].weight
            for k in 1:3
                values[i,j,k] = qd_points_1i[j].value[k].value
                cart[i,j,k] = qd_points_1i[j].point.cart[k]
            end
        end
    end
    # weights = move(backend, weights)
    # values = move(backend, values)
    # cart = move(backend, cart)

    return weights, values, cart
end

function schedule_kernel!(
        backend, 
        length_return_matrix, elements_length_tuple,
        assembly_gpu_data, 
        biop, should_calc, qd, floatType, store,
        time_table, time_to_store, pref_offet,
        elements_data,
        configuration,
        producers  = []
    )

    time = @elapsed begin
        length_1 = 3
        length_2 = 4
        womps_weights, womps_values, womps_cart = load_data(backend, floatType, elements_length_tuple[1], qd.tpoints, pref_offet,  length_1)
        wimps_weights, wimps_values, wimps_cart = load_data(backend, floatType, elements_length_tuple[2], qd.bpoints, 0, length_2)
        womps_weights, womps_values, womps_cart = move(backend, womps_weights), move(backend, womps_values), move(backend, womps_cart)
        wimps_weights, wimps_values, wimps_cart = move(backend, wimps_weights), move(backend, wimps_values), move(backend, wimps_cart)
        wimps_and_womps = [womps_weights, womps_values, womps_cart, wimps_weights, wimps_values, wimps_cart]
        # KernelAbstractions.synchronize(backend)
    end
    Threads.atomic_add!(time_table[1,1], time)
    # time_table[1,1] += time

    GC.@preserve womps_weights womps_values womps_cart wimps_weights wimps_values wimps_cart begin
    
        case = 3
        if typeof(configuration["writeBackStrategy"]) == GpuWriteBackTrueInstance
            result_1 = create_results_matrix_gpu(backend, length_return_matrix, elements_length_tuple, configuration["writeBackStrategy"], configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"])
            time_to_calc_double = @elapsed begin
                doubleQuadRule_generic_3d_gpu_outside_loop!(result_1,
                    assembly_gpu_data,
                    biop,
                    wimps_and_womps, 
                    0, 0,
                    time_table, 1, elements_length_tuple,
                    elements_data, floatType,
                    configuration,
                    should_calc
                )
                KernelAbstractions.synchronize(backend)
            end
            print(time_to_calc_double, ", ")

            # time_to_store += @elapsed begin

                # write_to_compact_matrix(result_1, store, length_return_matrix, size_qrule, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes)
            # end
        elseif case == 2
            GPU_budget = 7 * GiB
            amount_of_producers = 8#Threads.nthreads()
            amount_of_consumers = 4

            GPU_spent_by_data = (length_1 * elements_length_tuple[1] + 2 * 3 * length_1 * elements_length_tuple[1]
                                + length_2 * elements_length_tuple[2] + 2 * 3 * length_2 * elements_length_tuple[2]) * sizeof(floatType) #in bytes
            GPU_spent_by_should_calc = 1 * elements_length_tuple[1] * elements_length_tuple[2]
            GPU_budget -= (GPU_spent_by_data + GPU_spent_by_should_calc) * amount_of_producers

            size_submatrix = ceil(Int64, sqrt(GPU_budget / (amount_of_producers * 9 * sizeof(ComplexF64))))   # <=> GPU_budget =  2 * size_submatrix^2 * 9 * size(ComplexF64) 
            blocks_x = ceil(Int64, elements_length_tuple[1] / size_submatrix)
            blocks_y = ceil(Int64, elements_length_tuple[2] / size_submatrix)
            

            results = KernelAbstractions.allocate(backend, ComplexF64, (size_submatrix, size_submatrix, 9))

            for j in 1:blocks_y
                for i in 1:blocks_x
                    ndrange = [size_submatrix, size_submatrix]
                    if i == blocks_x
                        ndrange[1] = elements_length_tuple[1] - (i - 1) * size_submatrix
                    end
                    if j == blocks_y
                        ndrange[2] = elements_length_tuple[2] - (j - 1) * size_submatrix
                    end
                    ndrange = Tuple(ndrange)

                    curr_offsets = ((i - 1) * size_submatrix, (j - 1) * size_submatrix)

                    doubleQuadRule_generic_3d_gpu_outside_loop!(results,
                        test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values,
                        biop,
                        womps_weights, wimps_weights, 
                        womps_values, wimps_values, 
                        womps_cart, wimps_cart, 
                        (i - 1) * size_submatrix, (j - 1) * size_submatrix,
                        InstancedoubleQuadRuleGpuStrategyShouldCalculate, writeBackStrategy,
                        time_table, 1, ndrange,
                        test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatType,
                        should_calc
                    )


                    result_cpu = Array(results)
                    write_to_compact_matrix(result_cpu, store, length_return_matrix, ndrange, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, curr_offsets)
                end
            end

        



            


            

            
        elseif case == 3
            GPU_budget = configuration["GPU_budget_pipeline_result"]
            amount_of_producers = 3

            GPU_spent_by_data = (length_1 * elements_length_tuple[1] + 2 * 3 * length_1 * elements_length_tuple[1]
                                + length_2 * elements_length_tuple[2] + 2 * 3 * length_2 * elements_length_tuple[2]) * sizeof(floatType) #in bytes
            GPU_spent_by_should_calc = 1 * elements_length_tuple[1] * elements_length_tuple[2]

            GPU_budget -= GPU_spent_by_data + GPU_spent_by_should_calc

            size_submatrix = ceil(Int64, sqrt(GPU_budget / (amount_of_producers * 9 * sizeof(ComplexF64))))
            blocks_x = ceil(Int64, elements_length_tuple[1] / size_submatrix)
            blocks_y = ceil(Int64, elements_length_tuple[2] / size_submatrix)
            blocks = blocks_x

            # ch_store = Channel{Tuple{Array{ComplexF64, 3}, Tuple{Int64, Int64}, Tuple{Int64, Int64}, Int64, Int64}}(blocks_x * blocks_y)  
            ch_data = Channel{Tuple{Int64, Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}}(blocks_x * blocks_y)  

            # push!(ch_store_array, ch_store)
            time = @elapsed begin
                for i in 1:amount_of_producers
                    p = Threads.@spawn producer(ch_data, backend,
                        assembly_gpu_data,
                        biop,
                        wimps_and_womps, 
                        time_table, should_calc, size_submatrix,

                        store, length_return_matrix, blocks,
                        elements_data,
                        floatType, configuration
                    )
                    push!(producers, p)
                end
                for j in 1:blocks_y
                    for i in 1:blocks_x
                        ndrange = [size_submatrix, size_submatrix]
                        if i == blocks_x
                            ndrange[1] = elements_length_tuple[1] - (i - 1) * size_submatrix
                        end
                        if j == blocks_y
                            ndrange[2] = elements_length_tuple[2] - (j - 1) * size_submatrix
                        end
                        ndrange = Tuple(ndrange)

                        curr_offsets = ((i - 1) * size_submatrix, (j - 1) * size_submatrix)
                        item = (i, j, ndrange, curr_offsets)
                        put!(ch_data, item)
                    end
                end

                
                close(ch_data)
            
            
                for p in producers
                    wait(p)
                end
            end
            # @show time
        end
    end
end



