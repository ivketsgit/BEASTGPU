include("assamble_chunk_gpu.jl") 
include("utils/create_pinned_array_if_posible.jl")
include("utils/copy_to_CPU.jl")
using Base.Threads



function assemble_gpu(operator::BEAST.AbstractOperator, test_functions, trial_functions, config, writeBackStrategy::GpuWriteBackFalseInstance,;
    storage_policy = Val{:bandedstorage},
    threading = BEAST.Threading{:single},
    # long_delays_policy = LongDelays{:compress},
        quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions))

    Z_real, Z_imag, store = allocatestorage(operator, test_functions, trial_functions, storage_policy)
    # Z_, store = allocatestorage(operator, test_functions, trial_functions, storage_policy)

    assemble_gpu!(operator, test_functions, trial_functions, config, store, threading; quadstrat, split)

    real_part = Z_real()
    imag_part = Z_imag
    # Z_()
    return real_part + imag_part *im
end

function assemble_gpu(operator::BEAST.AbstractOperator, test_functions, trial_functions, config, writeBackStrategy::GpuWriteBackTrueInstance;
    storage_policy = Val{:bandedstorage},
    threading = BEAST.Threading{:single},
    # long_delays_policy = LongDelays{:compress},
        quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions))
    
    backend = config.backend

    
    result_cpu = Array{Float64}(undef, 1)  
    result_complex = Array{ComplexF64}(undef, 1)
    result_complex_array = []
    complex_array = KernelAbstractions.allocate(backend, ComplexF64, size(result_cpu))
    chunk_size = 1024 * 1024 * 10
    # nthreads = Int(round(Threads.nthreads() / 2))
    nthreads = Threads.nthreads()
    # task_allocate = Threads.@spawn begin
        if config.makeCompexWithGPU == true
            result_cpu = Array{ComplexF64}(undef, numfunctions(test_functions), numfunctions(trial_functions))
            

            # time_pinned = @elapsed begin
            #     result_complex = pinned_arr(result_cpu, backend)
            # end
            # @show time_pinned

            
            # time_pinned = @elapsed begin
            #     chunk_size = Int(ceil(prod(size(result_cpu))/100))
            #     for n in 1:nthreads
            #         push!(result_complex_array, pinned_arr(Array{ComplexF64}(undef, chunk_size), backend))
            #     end
            #     # config["pinned_buffers"] = result_complex_array
            #     # result_complex = pinned_arr(result_cpu, backend)
            # end
            # @show time_pinned
            # time_pinned = @elapsed begin
            #     result_complex = pinned_arr(result_cpu, backend)
            # end
            # @show time_pinned
            complex_array = KernelAbstractions.allocate(backend, ComplexF64, size(result_cpu))

            

        else
            result_cpu = Array{Float64}(undef, 2, numfunctions(test_functions), numfunctions(trial_functions))
            result_cpu = pinned_arr(result_cpu, backend)
        end
    # end

    assemble_gpu!(operator, test_functions, trial_functions, config, nothing, threading; quadstrat, split)


    gpu_array = gpu_results_cache[1]
    GiB = prod(size(gpu_array)) * sizeof(Float64) / 2^30
    
    if config.makeCompexWithGPU == true
        time_make_complex = @elapsed begin
            make_complex(backend)(complex_array, gpu_array, ndrange = (numfunctions(test_functions), numfunctions(trial_functions)))
            KernelAbstractions.synchronize(backend)
        end
        @show time_make_complex

        # time_to_transfer_with_copy = @elapsed begin
        #     copyto!(result_complex, complex_array)
        # end
        # @show time_to_transfer_with_copy

        
        
        
        time_to_transfer_with_copy = @elapsed begin
        #     # result_profile = CUDA.@profile trace=true copyto!(result_complex, complex_array)
            chunk_size = Int(ceil(prod(size(result_cpu))/100))
            result_cpu = copy_to_CPU(result_cpu, complex_array, backend, ComplexF64, chunk_size, config)

            # result_cpu = copy_to_CPU(result_cpu, complex_array, backend, ComplexF64, Int(round(1024 * 1024 * 100 * 1.5)), config)

        

            # rows, cols = size(result_cpu)
            # complex_array_flat = reshape(complex_array, :)  
            # result_cpu_flat = reshape(result_cpu, :)

            
            # N = prod(size(result_cpu))
        #     # for i in 1:chunk_size:N
        #     #     this_chunk = min(chunk_size, N - i + 1)
        #     #     copyto!(result_complex_1, 1, complex_array_flat, i, this_chunk)      # GPU -> pinned buffer
        #     #     copyto!(result_cpu_flat, i, result_complex_1, 1, this_chunk)        # pinned buffer -> final array
        #     # end

            # channel = Channel{Tuple{Int, Int}}(100)
            # task = []
            # for n in 1:nthreads
            #     t = Threads.@spawn worker(result_cpu_flat, result_complex_array[n], complex_array_flat, channel)
            #     push!(task, t)
            # end

            # for i in 1:chunk_size:N
            #     this_chunk = min(chunk_size, N - i + 1)
            #     put!(channel, (i, this_chunk))
            # end
            # close(channel)

            # for n in 1:nthreads
            #     wait(task[n])
            # end

            # result_cpu = reshape(result_cpu_flat, rows, cols)
        end
        @show time_to_transfer_with_copy
        println("GiB/s = ", GiB / time_to_transfer_with_copy)

        
    else
        
        time_to_transfer_with_copy = @elapsed begin
            copyto!(result_cpu, gpu_array)
        end
        @show time_to_transfer_with_copy
        println("GiB/s = ", GiB / time_to_transfer_with_copy)

        time_make_complex = @elapsed begin
            result_cpu = complex.(view(result_cpu, 1, :, :), view(result_cpu, 2, :, :))
        end
    end

    empty!(gpu_results_cache)


    if config.timeLogger !== nothing
        log_time(config.timeLogger, "transfer results to CPU", time_to_transfer_with_copy)
        log_time(config.timeLogger, "create results as complex numbers", time_make_complex)
    end

    return result_cpu
    # return result_complex
end



@kernel function make_complex(complex_array, @Const(float_array))
    i, j = @index(Global, NTuple)
    complex_array[i, j] = Complex(float_array[1, i, j], float_array[2, i, j])
end

function assemble_gpu!(operator::BEAST.Operator, test_functions::BEAST.Space, trial_functions::BEAST.Space, config,
    store, threading::Type{BEAST.Threading{:single}};
    quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    split = false)
    
    assemblechunk_gpu!(operator, test_functions, trial_functions, config, store; quadstrat)
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