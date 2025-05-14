using CUDA

@show CUDA.device()

device = CUDA.device()
println("Device ID: ", device)
println("Device Name: ", CUDA.name(device))

for i in 0:CUDA.device_count()-1
    dev = CuDevice(i)
    println("Device $i: ", CUDA.name(dev))
end