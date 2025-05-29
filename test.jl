using CUDA
using BenchmarkTools

data = rand(Float32, 2^28)
@show Base.format_bytes(sizeof(data))

 @belapsed println("test")

# println("Host-to-GPU w/o allocating")

# time = @belapsed begin dA = CuArray(data) end
# @show Base.format_bytes(sizeof(data) / time) * "/s"


# println("Host-to-GPU w allocation")

dA = CuArray{Float32}(undef, length(data))
# time = @belapsed copyto!(dA, data)
# @show Base.format_bytes(sizeof(data) / time) * "/s"


# println("GPU-to-Host w/o allocating")

# time = @belapsed begin b = Array(dA) end
# @show Base.format_bytes(sizeof(data) / time) * "/s"

let 
    println("GPU-to-Host w allocation")

    b = similar(data)
    time = @belapsed copyto!(b,dA)
    @show Base.format_bytes(sizeof(b) / time) * "/s"

    println("GPU-to-Host w allocation 2")

    time = @belapsed begin
    b = similar(data)
    copyto!(b,dA)
    end
    @show Base.format_bytes(sizeof(b) / time) * "/s"

    println("GPU-to-Host w allocation + modification")
    b = similar(data)
    time = @belapsed begin
    fill!(b,0.0)
    copyto!(b,dA)
    end
    @show Base.format_bytes(sizeof(b) / time) * "/s"


    # @assert b==data
end








println("GPU-to-Host w allocation")

b = similar(data)
let time = @elapsed copyto!(b,dA) 
    println(Base.format_bytes(sizeof(data) / time) , "/s")
end

println("GPU-to-Host w allocation 2")

let time = @elapsed begin
        b = similar(data)
        copyto!(b,dA)
    end
    println(Base.format_bytes(sizeof(data) / time) , "/s")
end









