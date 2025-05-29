function copy_to_CPU(CPU_array, GPU_array, backend, type, chunk_size)
    nthreads = Threads.nthreads()

    buffers = []
    # wanted_size = 1024*1024 * 100 * 4
    # chunk_size = Int(round(wanted_size / sizeof(type)))
    # chunk_size = 1024*1024
    
    time_allocation = @elapsed begin
        for n in 1:nthreads
            push!(buffers, pinned_arr(Array{type}(undef, chunk_size), backend))
        end
    end
    @show time_allocation

    time_transfer = @elapsed begin
        rows, cols = size(GPU_array)
        GPU_array_flat = reshape(GPU_array, :)  
        CPU_array_flat = reshape(CPU_array, :)

        N = prod(size(CPU_array))
        channel = Channel{Tuple{Int, Int}}(10000)
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
    for (i, this_chunk) in channel
        # println(i)
        copyto!(buffer, 1, GPU_array_flat, i, this_chunk)      # GPU -> pinned buffer
        copyto!(CPU_array_flat, i, buffer, 1, this_chunk)        # pinned buffer -> final array
    end
end
