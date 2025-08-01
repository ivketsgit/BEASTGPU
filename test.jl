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


GC.gc()

t = @elapsed begin
    CUDA.pin(Array{ComplexF64}(undef, Int(round(38400*0.5)),38400))
end 
@show t
GC.gc()