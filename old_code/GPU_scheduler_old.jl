# mutable struct Queue{T}
#     data :: Vector{T}
#     finished :: Bool
#     lock :: ReentrantLock
#     cond :: Base.Condition
# end

# Queue() = Queue{Any}(Vector(), false, ReentrantLock(), Base.Condition())

# function enqueue!(q::Queue, item)
#     lock(q.lock) do
#         push!(q.data, item)
#         notify(q.cond)  
#     end
# end

# function dequeue!(q::Queue)
#     lock(q.lock) do
#         return isempty(q.data) ? nothing : popfirst!(q.data)
#     end
# end

# function is_empty(q::Queue)
#     lock(q.lock) do
#         return isempty(q.data)
#     end
# end

# function is_finished(q::Queue)
#     lock(q.lock) do
#         q.finished
#     end
# end

# function set_finished(q::Queue)
#     lock(q.lock) do
#         q.finished = true
#     end
# end

# function process_queue(q::Queue)
#     try
#         while true
#             lock(q.lock) do
#                 while is_empty(q) && !(q.finished)  # Wait if queue is empty and not stopping
#                     wait(q.cond)
#                 end
#             end

#             item = dequeue!(q)
#             if item !== nothing
#                 # println("Processing: ", item)
#                 result_cpu, prev_range, prev_offsets = item
#                 write_to_compact_matrix(result_cpu, store, length_return_matrix, prev_range, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, prev_offsets)
#             elseif q.finished
#                 print("\n")
#                 println("Queue is empty, stopping.")
#                 break
#             end
#         end
#     catch err
#         println("Error in process_queue: ", err)
#         println("Stacktrace: ", stacktrace(catch_backtrace()))
#     end
# end



        # result_prev_cpu = Array{ComplexF64}(undef, size_submatrix, size_submatrix, 9)
        
        # q = Queue()
        # task = Threads.@spawn process_queue(q)

        # finished = zeros(blocks_x * blocks_y)
        # 
        # n_consumers = 12
        # tasks = []
        # for i in 1:n_consumers
        #     push!(tasks, Threads.@spawn consumer(ch, store, length_return_matrix, writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, time_to_store_ref))
        # end


        

        # ptr_curr_result = result_1
        # ptr_prev_result = result_2
        # ptr_temp = nothing
        # prev_range = nothing
        # curr_range = (0,0)
        # prev_offsets = nothing
        # curr_offsets = (0,0)



        # result_curr_cpu = Array{ComplexF64}(undef, size_submatrix, size_submatrix, 9)
        # # result_prev_cpu = Array{ComplexF64}(undef, size_submatrix, size_submatrix, 9)
        # for j in 1:blocks
        #     for i in 1:blocks
        #         # @show i, j
        #         ndrange = [size_submatrix, size_submatrix]
        #         if i == blocks
        #             ndrange[1] = size_qrule - (i - 1) * size_submatrix
        #         end
        #         if j == blocks
        #             ndrange[2] = size_qrule - (j - 1) * size_submatrix
        #         end
        #         ndrange = Tuple(ndrange)
        #         # @show ndrange

        #         curr_range = ndrange
        #         curr_offsets = ((i - 1) * size_submatrix, (j - 1) * size_submatrix)
                

        #         doubleQuadRule_generic_3d_gpu_outside_loop!(ptr_curr_result,
        #             test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values,
        #             size_qrule,
        #             biop,
        #             womps_weights, wimps_weights, 
        #             womps_values, wimps_values, 
        #             womps_cart, wimps_cart, 
        #             (i - 1) * size_submatrix, (j - 1) * size_submatrix,
        #             InstancedoubleQuadRuleGpuStrategyShouldCalculate, writeBackStrategy,
        #             time_table, 1, ndrange, should_calc
        #         )


                
        #         if !(i == 1 && j == 1)
        #             result_cpu = Array(ptr_prev_result)
        #             # KernelAbstractions.copyto!(backend_cpu, result_curr_cpu, ptr_prev_result)

        #         end
                
        #         KernelAbstractions.synchronize(backend)
                
        #         if !(i == 1 && j == 1)
        #             # item = (copy(result_curr_cpu), prev_range, prev_offsets)
        #             item = (result_cpu, prev_range, prev_offsets)
        #             put!(ch, item)
        #         end

        #         ptr_temp = ptr_curr_result
        #         ptr_curr_result = ptr_prev_result
        #         ptr_prev_result = ptr_temp

        #         prev_range = curr_range
        #         prev_offsets = curr_offsets
        #     end
        #     print(".")
        # end
        # # item = (copy(result_curr_cpu), prev_range, prev_offsets)
        # result_cpu = Array(ptr_prev_result)
        # item = (result_cpu, prev_range, prev_offsets)
        # put!(ch, item)
        # close(ch)  
        # wait(t)      
        


        # result_prev_cpu = Array{ComplexF64}(undef, size_submatrix, size_submatrix, 9)
        # prev_prev_task = nothing
        # prev_task = nothing





































        # for j in 1:blocks
        #     for i in 1:blocks
        #         # @show i, j
        #         ndrange = [size_submatrix, size_submatrix]
        #         if i == blocks
        #             ndrange[1] = size_qrule - (i - 1) * size_submatrix
        #         end
        #         if j == blocks
        #             ndrange[2] = size_qrule - (j - 1) * size_submatrix
        #         end
        #         ndrange = Tuple(ndrange)
        #         # @show ndrange

        #         curr_range = ndrange
        #         curr_offsets = ((i - 1) * size_submatrix, (j - 1) * size_submatrix)
                
        #         doubleQuadRule_generic_3d_gpu_outside_loop!(ptr_curr_result,
        #             test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values,
        #             size_qrule,
        #             biop,
        #             womps_weights, wimps_weights, 
        #             womps_values, wimps_values, 
        #             womps_cart, wimps_cart, 
        #             (i - 1) * size_submatrix, (j - 1) * size_submatrix,
        #             InstancedoubleQuadRuleGpuStrategyShouldCalculate, writeBackStrategy,
        #             time_table, 1, ndrange, should_calc
        #         )

        #         KernelAbstractions.copyto!(backend_cpu, result_curr_cpu, ptr_prev_result)
        #         item = (copy(result_curr_cpu), ndrange, curr_offsets)
        #         put!(ch, item)

        #         # KernelAbstractions.pagelock!(backend, backend_cpu)
                
        #         # KernelAbstractions.synchronize(backend)
                
        #         # if !(i == 1 && j == 1)
        #         #     item = (copy(result_curr_cpu), prev_range, prev_offsets)
        #         #     put!(ch, item)
        #         # end

        #         ptr_temp = ptr_curr_result
        #         ptr_curr_result = ptr_prev_result
        #         ptr_prev_result = ptr_temp

        #         prev_range = curr_range
        #         prev_offsets = curr_offsets
        #     end
        #     print(".")
        # end
        # # item = (copy(result_curr_cpu), prev_range, prev_offsets)
        # # put!(ch, item)
        # close(ch)  
        # wait(t)      











        