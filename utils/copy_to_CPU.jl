function copy_to_CPU(CPU_array, GPU_array, backend, type, chunk_size, config)
    # nthreads = Int(round(Threads.nthreads() / 2))
    nthreads = Threads.nthreads()
    # chunk_size = 1024 * 1024 * 10
    

    buffers = []
    # wanted_size = 1024*1024 * 100 * 4
    # chunk_size = Int(round(wanted_size / sizeof(type)))
    # chunk_size = 1024*1024
    time_allocation = @elapsed begin
        # if config["makeCompexWithGPU"] == true
        #     buffers = config["pinned_buffers"]
        # else
            for n in 1:nthreads
                push!(buffers, pinned_arr(Array{type}(undef, chunk_size), backend))
            end
        # end
    end
    @show time_allocation

    time_transfer = @elapsed begin
        rows, cols = size(GPU_array)
        GPU_array_flat = reshape(GPU_array, :)  
        CPU_array_flat = reshape(CPU_array, :)

        N = prod(size(CPU_array))
        channel = Channel{Tuple{Int, Int}}(10000)
        gpu_copy_semaphore = Base.Semaphore(12)  
        tasks = []
        for n in 1:nthreads
            task = Threads.@spawn worker(CPU_array_flat, buffers[n], GPU_array_flat, channel)
            push!(tasks, task)
        end

        for i in 1:chunk_size:N
            this_chunk = min(chunk_size, N - i + 1)
            put!(channel, (i, this_chunk))
        end
        close(channel)

        for n in 1:nthreads
            wait(tasks[n])
        end
    end
    @show time_transfer

    CPU_array = reshape(CPU_array_flat, rows, cols)
    return CPU_array
end


function worker(CPU_array_flat, buffer, GPU_array_flat, channel)
    # gpu_to_buffer_time = 0.0
    # buffer_to_cpu_time = 0.0
    for (i, this_chunk) in channel
        # gpu_to_buffer_time += @elapsed copyto!(buffer, 1, GPU_array_flat, i, this_chunk)      # GPU -> pinned buffer
        # buffer_to_cpu_time += @elapsed copyto!(CPU_array_flat, i, buffer, 1, this_chunk)        # pinned buffer -> final array
        unsafe_copyto!(buffer, 1, GPU_array_flat, i, this_chunk)      # GPU -> pinned buffer
        unsafe_copyto!(CPU_array_flat, i, buffer, 1, this_chunk)        # pinned buffer -> final array
    end
    # @show gpu_to_buffer_time
    # @show buffer_to_cpu_time
end


function worker(CPU_array_flat, buffer, GPU_array_flat, channel, gpu_copy_semaphore)
    gpu_to_buffer_time = 0.0
    buffer_to_cpu_time = 0.0
    for (i, this_chunk) in channel
        Base.acquire(gpu_copy_semaphore)
        try
            copyto!(buffer, 1, GPU_array_flat, i, this_chunk)     # GPU -> pinned buffer
        finally
            # Always release the permit
            Base.release(gpu_copy_semaphore)
        end
        copyto!(CPU_array_flat, i, buffer, 1, this_chunk)     # pinned buffer -> CPU
    end
end
