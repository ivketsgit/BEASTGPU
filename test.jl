using CUDA
using BenchmarkTools

# @show CUDA.device()

# device = CUDA.device()
# println("Device ID: ", device)
# println("Device Name: ", CUDA.name(device))

# for i in 0:CUDA.device_count()-1
#     dev = CuDevice(i)
#     println("Device $i: ", CUDA.name(dev))
# end



t = @elapsed begin
    # CUDA.pin(Array{ComplexF64}(undef, Int(round(38400/1.5)),38400))
    # CUDA.pin(Array{ComplexF64}(undef, 38400, 38400))
    # CuArray()

    CUDA.pin(Array{ComplexF64}(undef, Int(round(38400)),38400))
end 
@show t
GC.gc()