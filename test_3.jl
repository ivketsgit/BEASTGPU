using CUDA
let 
    
    t = time_elapsed = @elapsed begin
        arr = Array{ComplexF64}(undef, (38400, 38400))
        CUDA.pin(arr)
    end
    @show t

    t = time_elapsed = @elapsed begin
        arr = Array{ComplexF64}(undef, (38400, Int(38400/2)))
        arr2 = Array{ComplexF64}(undef, (38400, Int(38400/2)))
        CUDA.pin(arr)
        CUDA.pin(arr2)
    end
    @show t

    t = time_elapsed = @elapsed begin
        arr = Array{ComplexF64}(undef, (38400, Int(38400/2)))
        CUDA.pin(arr)
    end
    @show t
end