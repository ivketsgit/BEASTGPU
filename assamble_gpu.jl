include("assamble_chunk_gpu.jl") 
using Base.Threads


function log_time(time_logger, key::String, value)
    if haskey(time_logger, key)
        push!(time_logger[key], value)
    else
        time_logger[key] = [value]
    end
end



function assemble_gpu(operator::BEAST.AbstractOperator, test_functions, trial_functions, configuration;
    storage_policy = Val{:bandedstorage},
    threading = BEAST.Threading{:single},
    # long_delays_policy = LongDelays{:compress},
        quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions))
    print(".")

    Z_real, Z_imag, store = allocatestorage(operator, test_functions, trial_functions, storage_policy)
    # Z_, store = allocatestorage(operator, test_functions, trial_functions, storage_policy)
    
    split = false
    assemble_gpu!(operator, test_functions, trial_functions, configuration, store, threading; quadstrat, split)
    if typeof(configuration["writeBackStrategy"]) == GpuWriteBackTrueInstance
        # CUDA.unsafe_free!(gpu_results_cache[1])
        # CUDA.synchronize()
        # CUDA.reclaim()
        # return
        # time_read_out_matrix = @elapsed begin

            time_to_transfer_originel = @elapsed begin
                result_cpu = Array(gpu_results_cache[1])
            end
            @show time_to_transfer_originel

            # time_to_transfer_results_2 = @elapsed begin
                time_gpu_array = @elapsed begin
                    gpu_array = gpu_results_cache[1]
                end
                @show time_gpu_array
                
                time_allocate = @elapsed begin
                    # result_cpu = Array{Float64}(undef, (2, 38402, 38402))
                    result_cpu = Array{Float64}(undef, size(gpu_array))
                end
                @show time_allocate

                
                time_to_transfer_with_copy = @elapsed begin
                    copyto!(result_cpu, gpu_array)
                end
                @show time_to_transfer_with_copy

                
                time_to_transfer_with_copy_KA_1 = @elapsed begin
                    KernelAbstractions.copyto!(CUDABackend(), result_cpu, gpu_array)
                end
                @show time_to_transfer_with_copy_KA_1

                
                time_to_transfer_with_copy_KA_2 = @elapsed begin
                    KernelAbstractions.copyto!(CPU(), result_cpu, gpu_array)
                end
                @show time_to_transfer_with_copy_KA_2

            # end
            # @show time_to_transfer_results_2

            time_make_complex = @elapsed begin
                N, M = size(result_cpu, 2), size(result_cpu, 3)
                result_complex = Matrix{ComplexF64}(undef, N, M)

                @threads for j in 1:M
                    for i in 1:N
                        result_complex[i, j] = complex(result_cpu[1, i, j], result_cpu[2, i, j])
                    end
                end
                # result_complex = complex.(view(result_cpu, 1, :, :), view(result_cpu, 2, :, :))
            end
            # @show time_to_transfer_results, time_make_complex

            empty!(gpu_results_cache)
            GC.gc()
            #CUDA.memory_status() 
            # @info "time to transfer results to CPU= $time_to_transfer_results"
            # @info "time to create results as complex numbers = $time_make_complex"
            if isdefined(Main, :time_logger)
                
                # log_time(time_logger, "transfer results to CPU", time_to_transfer_originel)
                log_time(time_logger, "create results as complex numbers", time_make_complex)
            end
            empty!(gpu_results_cache)
            return result_complex
        # end
    #     # @show time_read_out_matrix
    #     # time_read_out_matrix = @elapsed begin
    #     #     result_gpu = gpu_results_cache[1]
    #     #     result_cpu = complex.(Array(result_gpu[1, :, :]), Array(result_gpu[2, :, :]))
    #     # end
    #     # @show time_read_out_matrix
    #     # time_read_out_matrix = @elapsed begin
    #     #     result_cpu_raw = Array(gpu_results_cache[1])
    #     #     M, N = size(result_cpu_raw, 2), size(result_cpu_raw, 3)
    #     #     result_cpu = Array{ComplexF32}(undef, M, N)  # or ComplexF64 depending on original
    #     #     @inbounds for i in 1:M, j in 1:N
    #     #         result_cpu[i, j] = Complex(result_cpu_raw[1, i, j], result_cpu_raw[2, i, j])
    #     #     end 
    #     # end
    #     # @show time_read_out_matrix
    #     empty!(gpu_results_cache)
    #     return result_cpu
    end
    real_part = Z_real()
    imag_part = Z_imag
    # Z_()
    return real_part + imag_part *im
end


function assemble_gpu!(operator::BEAST.Operator, test_functions::BEAST.Space, trial_functions::BEAST.Space, configuration,
    store, threading::Type{BEAST.Threading{:single}};
    quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    split = false)
    
    assemblechunk_gpu!(operator, test_functions, trial_functions, configuration, store; quadstrat)
end

# function assemble_gpu!(operator::BEAST.Operator, test_functions::BEAST.Space, trial_functions::BEAST.Space,
#     store, threading::Type{BEAST.Threading{:single}};
#     quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
#     split = true)

#     GPU_budget = 10^30

#     test_elements_length = x 
#     trial_elements_length = y

#     @show test_functions
#     @show numfunctions(test_functions)



#     # (length_1 * elements_length_tuple[1] + 2 * 3 * length_1 * elements_length_tuple[1]
#     #  + length_2 * elements_length_tuple[2] + 2 * 3 * length_2 * elements_length_tuple[2]) * sizeof(Float64)

#     assemblechunk_gpu!(operator, test_functions, trial_functions, store; quadstrat)
# end

function assemble_gpu!(operator::BEAST.Operator, test_functions::BEAST.Space, trial_functions::BEAST.Space,
    store, threading::Type{BEAST.Threading{:multi}};
    quadstrat=defaultquadstrat(operator, test_functions, trial_functions))

    P = 8#Threads.nthreads()
    numchunks = P
    @assert numchunks >= 1
    splits = [round(Int,s) for s in range(0, stop=numfunctions(test_functions), length=numchunks+1)]

    Threads.@threads for i in 1:P
        lo, hi = splits[i]+1, splits[i+1]
        lo <= hi || continue
        test_functions_p = BEAST.subset(test_functions, lo:hi)

        store1 = BEAST._OffsetStore(store, lo-1, 0)
        assemblechunk_gpu!(operator, test_functions_p, trial_functions, store1, quadstrat=quadstrat)
    end 
end


function allocatestorage(operator::BEAST.AbstractOperator, test_functions, trial_functions,
    storage_trait=nothing, longdelays_trait=nothing)

    T = promote_type(
        scalartype(operator)       ,
        scalartype(test_functions) ,
        scalartype(trial_functions),
    )
    Z_real = Matrix{Float64}(undef,
        numfunctions(test_functions),
        numfunctions(trial_functions),
    )
    Z_imag = Matrix{Float64}(undef,
        numfunctions(test_functions),
        numfunctions(trial_functions),
    )
    # Z = Matrix{ComplexF64}(undef,
    #     numfunctions(test_functions),
    #     numfunctions(trial_functions),
    # )
    # fill!(Z, 0)
    fill!(Z_real, 0)
    fill!(Z_imag, 0)
    # store(v,m,n) = (Z[m,n] += v)
    store(v, m, n) = begin
        @atomic Z_real[m, n] += real(v)
        @atomic Z_imag[m, n] += imag(v)
        # Z[m, n] += v
    end
    return ()->Z_real, Z_imag, store
    # return ()->Z, store
end



# defaultquadstrat(op::HelmholtzOperator2D, tfs, bfs) = DoubleNumQStrat(4,3)
# struct DoubleNumQStrat{R}
#     outer_rule::R
#     inner_rule::R
# end
# function quadrule(operator::IntegralOperator,
#     local_test_basis, local_trial_basis,
#     test_id, test_element, trial_id, trial_element,
#     quad_data, qs::DoubleNumQStrat)

#     test_quad_rules  = quad_data[1]
#     trial_quad_rules = quad_data[2]

#     DoubleQuadRule(
#         test_quad_rules[1,test_id],
#         trial_quad_rules[1,trial_id]
#     )
# end
# struct DoubleQuadRule{P,Q}
#     outer_quad_points::P
#     inner_quad_points::Q
# end   