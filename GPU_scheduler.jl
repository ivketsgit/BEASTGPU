include("doubleQuadRuleMainCase.jl")

@inline function load_data_3(backend, floatType, elements_length_tuple, qd, pref_offet, length_1, length_2, length_3 = 3)
    t_1 = @elapsed begin
        womps_weights = Array{floatType}(undef, elements_length, length_1)
        womps_values = Array{floatType}(undef, elements_length, length_1, length_3)
        womps_cart =   Array{floatType}(undef, elements_length, length_1, length_3)
        @inbounds for i in 1:elements_length
            qd_points_1i = qd_points[1,i + pref_offet]
            @inbounds for j in 1:length_1
                qdj = qd_points_1i[j]
                womps_weights[i,j] = qdj.weight
                @inbounds for k in 1:3
                    womps_values[i,j,k] = qdj.value[k].value
                end
                womps_cart[i,j,:] .= qdj.point.cart[:]
            end
        end
    end
    @show t_1

    
    t_2 = @elapsed begin
        womps_weights = move(backend, womps_weights)
        womps_values = move(backend, womps_values)
        womps_cart = move(backend, womps_cart)
    end
    @show t_2

    
    t_3 = @elapsed begin
        wimps_weights = Array{floatType}(undef, elements_length, length_2)
        wimps_values = Array{floatType}(undef, elements_length, length_2, length_3)
        wimps_cart =   Array{floatType}(undef, elements_length, length_2, length_3)
        @inbounds for i in 1:elements_length
            qd_points_1i = qd_points[1,i + pref_offet]
            @inbounds for j in 1:length_2
                qdj = qd_points_1i[j]
                wimps_weights[i,j] = qdj.weight
                @inbounds for k in 1:3
                    wimps_values[i,j,k] = qdj.value[k].value
                end
                wimps_cart[i,j,:] .= qdj.point.cart[:]
            end
        end
    end
    @show t_3

    
    t_4 = @elapsed begin
        wimps_weights = move(backend, wimps_weights)
        wimps_values = move(backend, wimps_values)
        wimps_cart = move(backend, wimps_cart)
    end
    @show t_4

    return womps_weights, womps_values, womps_cart, wimps_weights, wimps_values, wimps_cart
end


@inline function load_data(config, elements_length, qd_points, pref_offset, length_1, length_2 = 3)
    backend = config.backend
    type = config.floatType

    weights = Array{type}(undef, elements_length, length_1)
    values = Array{type}(undef, elements_length, length_1, 3)
    cart =   Array{type}(undef,elements_length, length_1, 3)
    
    @inbounds for i in 1:elements_length
        qd_points_1i = qd_points[1, i + pref_offset]
        for j in 1:length_1
            qdj = qd_points_1i[j]
            weights[i, j] = qdj.weight

            v = qdj.value
            values[i, j, 1] = v[1].value
            values[i, j, 2] = v[2].value
            values[i, j, 3] = v[3].value

            cart_ = qdj.point.cart
            cart[i, j, 1] = cart_[1]
            cart[i, j, 2] = cart_[2]
            cart[i, j, 3] = cart_[3]
        end
    end
    weights = move(backend, weights)
    values = move(backend, values)
    cart = move(backend, cart)
    return weights, values, cart
end

@inline function gether_wimps_and_womps(config, elementAssemblyData, qd, timingInfo)
    time = @elapsed begin
        length_1 = 3
        length_2 = 4
        
        elements_length_tuple = elementAssemblyData.elements_length_tuple
        womps_weights, womps_values, womps_cart = load_data(config, elements_length_tuple[1], qd.tpoints, elementAssemblyData.pref_offset,  length_1)
        wimps_weights, wimps_values, wimps_cart = load_data(config, elements_length_tuple[2], qd.bpoints, 0, length_2)
            
        wimps_and_womps = [womps_weights, womps_values, womps_cart, wimps_weights, wimps_values, wimps_cart]
    end
    Threads.atomic_add!(timingInfo.time_table[1,1], time)
    return wimps_and_womps

end

@inline function schedule_kernel!(elementAssemblyData,
        biop, quadrule_types_gpu, qd, store,
        timingInfo,
        config,
        writeBackStrategy::GpuWriteBackTrueInstance
    )
    wimps_and_womps = gether_wimps_and_womps(config, elementAssemblyData, qd, timingInfo)

    result_1 = create_results_matrix_gpu(config.backend, 
                                        elementAssemblyData.length_return_matrix, elementAssemblyData.elements_length_tuple, 
                                        config.writeBackStrategy, config.InstancedoubleQuadRuleGpuStrategyShouldCalculate)

    doubleQuadRuleMainCase!(result_1,
        elementAssemblyData,
        biop,
        wimps_and_womps, 
        (0, 0), elementAssemblyData.elements_length_tuple,
        timingInfo, 1,
        config,
        quadrule_types_gpu
    )
end

        
            

@inline function schedule_kernel!(elementAssemblyData,
        biop, quadrule_types_gpu, qd, store,
        timingInfo,
        config,
        writeBackStrategy::GpuWriteBackFalseInstance


        # backend, 
        # length_return_matrix, elements_length_tuple,
        # data, 
        # biop, quadrule_types_gpu, qd, floatType, store,
        # time_table, time_to_store, pref_offset,
        # config,
        # writeBackStrategy::GpuWriteBackFalseInstance
    )

    length_1 = 3
    length_2 = 4
    
    wimps_and_womps = gether_wimps_and_womps(config, elementAssemblyData, qd, timingInfo)

    GPU_budget = config.GPU_budget_pipeline_result
    amount_of_producers = config.amount_of_producers

    elements_length_tuple = elementAssemblyData.elements_length_tuple

    GPU_spent_by_data = (length_1 * elements_length_tuple[1] + 2 * 3 * length_1 * elements_length_tuple[1]
                        + length_2 * elements_length_tuple[2] + 2 * 3 * length_2 * elements_length_tuple[2]) * sizeof(config.floatType) #in bytes
    GPU_spent_by_should_calc = 1 * elements_length_tuple[1] * elements_length_tuple[2]

    GPU_budget -= GPU_spent_by_data + GPU_spent_by_should_calc
    

    size_submatrix = ceil(Int64, sqrt(GPU_budget / (amount_of_producers * 9 * sizeof(ComplexF64))))
    blocks_x = ceil(Int64, elements_length_tuple[1] / size_submatrix)
    blocks_y = ceil(Int64, elements_length_tuple[2] / size_submatrix)
    blocks = blocks_x

    ch_data = Channel{Tuple{Int64, Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}}(blocks_x * blocks_y)  

    time = @elapsed begin
        producers = []
        file_lock = ReentrantLock()
        for i in 1:amount_of_producers
            result_cpu_ = KernelAbstractions.allocate(CPU(), eltype(ComplexF64), (size_submatrix, size_submatrix, 9))
            result_cpu = pinned_arr(result_cpu_, config.backend)

            p = Threads.@spawn producer(ch_data,
                elementAssemblyData,
                biop,
                wimps_and_womps, 
                timingInfo, quadrule_types_gpu, size_submatrix,
                store, blocks,
                config, file_lock, result_cpu
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
end



function producer(ch_data::Channel,
                elementAssemblyData,
                biop,
                wimps_and_womps, 
                timingInfo, quadrule_types_gpu, size_submatrix,
                store, blocks,
                config, file_lock, result_cpu
    )
    backend = config.backend
    # result_cpu_ = KernelAbstractions.allocate(CPU(), eltype(ComplexF64), (size_submatrix, size_submatrix, 9))
    results = KernelAbstractions.allocate(backend, ComplexF64, size_submatrix, size_submatrix, 9)
    t1 = 0
    t2 = 0
    t3 = 0
    counter = 0
    GC.@preserve results begin
        for item in ch_data
            counter += 1
            t1 += @elapsed begin
                i, j, ndrange, curr_offsets = item

                doubleQuadRuleMainCase!(results,
                                    elementAssemblyData,
                                    biop,
                                    wimps_and_womps, 
                                    ((i - 1) * size_submatrix, (j - 1) * size_submatrix), ndrange,
                                    timingInfo, 1, 
                                    config,
                                    quadrule_types_gpu
                                )
            end
            t2 += @elapsed begin
                KernelAbstractions.copyto!(CPU(), result_cpu, results)
            end
            t3 += @elapsed begin
            # result_cpu_temp = Array(results)
            
            # result_cpu = Array(results)
            # write_to_compact_matrix(result_cpu, store, length_return_matrix, ndrange, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, curr_offsets)
            linear_index = (j - 1) * blocks + i
           
            if typeof(config.writeBackStrategy) == GpuWriteBackFalseInstance
                assembly_gpu_data = elementAssemblyData.assembly_data
                test_assembly_cpu_indexes = assembly_gpu_data[5]
                trial_assembly_cpu_indexes = assembly_gpu_data[6]
                write_to_compact_matrix(result_cpu, store, elementAssemblyData.length_return_matrix, ndrange, config.writeBackStrategy, config.InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, curr_offsets)
            end
            end
            # item = (result_cpu, ndrange, curr_offsets, i, j)
            # put!(ch_store, item)
            # println("produced ",i, " ",j)
        end
    end
    if hasfield(typeof(config), :gpu_schedular_print_filename)
        lock(file_lock) do
            open(config.gpu_schedular_print_filename, "a") do file
                println(file, "[",t1, ", ", t2, ", ", t3, ", ", counter, "],")
            end
        end
    end
end