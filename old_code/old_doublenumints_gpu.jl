using KernelAbstractions, CUDA, Atomix
using CUDA: CuArray, CUDAKernels

# include("C:/Users/Ian/.julia/dev/BEAST/zzz/doubleQuadRule_3d_gpu.jl")  
#include("../../zzz/strategy_sauterschwabints.jl")  

using .DoubleQuadRule_3d_gpu       
struct GPU_BackendWrapper
    backend_
end



@kernel function calc_j!(A, @Const(B), @Const(C))
    I, J = @index(Global, NTuple)
    A[I,J] = B[I] * C[J]
end

@kernel function calc_r!(A, @Const(B), @Const(C))
    I, J = @index(Global, NTuple)
    A[I,J] = B[I] - C[J]
end

@kernel function  calc_norm_2!(norm, @Const(x),@Const(y))
    I, J = @index(Global, NTuple)
    norm[I,J] = sqrt(x[I,J]^2 + y[I,J]^2)
end

@kernel function const_mul_vect!(A, @Const(c), @Const(B))
    I = @index(Global)
    A[I] = c * B[I]
end


function J_0_aprox(x)
    t = x^2*0.25
    return 1 + t * (-1 + t*0.25 * (1 + t * 0.11111111 * (-1 + t * 0.0625 * (1 + t * 0.04 ))))
end
function J_1_aprox(x)
    t = x^2*0.125
    return x/2 * (1 + t * (-1 + t*0.33333334 * (1 + t*0.16666667 * (-1 + t*0.1 * (1 + t*0.06666667)) ) ))
end
function Y_0_aprox(x)
    gamma = 0.5772157f0
    t = x^2*0.25
    return 2/pi*(log(ℯ,x/2) * J_0_aprox(x) + ((gamma) + t * ((-gamma + 1)+ t*0.25 * ((gamma - 1.5) + t*0.11111111 * ((-gamma + 1.8333334) + (gamma - 2.0833333) * t*0.0625)))))
end    
function Y_1_aprox(x)
    gamma = 0.5772157f0
    t = x^2*0.125 
    return (2 * (log(ℯ,x/2) * J_1_aprox(x) - 1/x) - x/2 * ((-2*gamma + 1) * 1 + t * ((2*gamma - 5/2)  + t/3 * ((-2*gamma + 10/3) + t/6 *((2*gamma - 47/12)  + (-2*gamma + 131/30) * t/10)))))/pi
end

function H_0_(x)
    t = x^2*0.25
    J = 1 + t * (-1 + t*0.25 * (1 + t * 0.11111111 * (-1 + t * 0.0625 * (1 + t * 0.04 ))))
    gamma = 0.5772157f0
    Y = 2/pi*(log(ℯ,x/2) * J + ((gamma) + t * ((-gamma + 1)+ t*0.25 * ((gamma - 1.5) + t*0.11111111 * ((-gamma + 1.8333334) + (gamma - 2.0833333) * t*0.0625)))))
    H = J - Y * im
    return H
end

function H_1_(x)
    t = x^2*0.125
    J = x/2 * (1 + t * (-1 + t*0.33333334 * (1 + t*0.16666667 * (-1 + t*0.1 * (1 + t*0.06666667)) ) ))
    gamma = 0.5772157f0
    Y = (2 * (log(ℯ,x/2) * J - 1/x) - x/2 * ((-2*gamma + 1) * 1 + t * ((2*gamma - 5/2)  + t/3 * ((-2*gamma + 10/3) + t/6 *((2*gamma - 47/12)  + (-2*gamma + 131/30) * t/10)))))/pi
    H = J - Y * im
    return H
end

@kernel function Hankel_type_2!(hankels_order_0, hankels_order_1, @Const(kr))
    I, J = @index(Global, NTuple)
    x = kr[I,J]
    hankels_order_0[I,J] = H_0_(x)
    hankels_order_1[I,J] = H_1_(x)
end

@kernel function green!(green, gradgreen_1, gradgreen_2,  @Const(hankels_order_0),  @Const(hankels_order_1), @Const(r_div_R_1), @Const(r_div_R_2), @Const(k))
    I, J = @index(Global, NTuple)
    green[I,J] = -im / 4 * hankels_order_0[I,J]
    ho1 = hankels_order_1[I,J]
    gradgreen_1[I,J] = k * im / 4 * ho1 * r_div_R_1[I,J]
    gradgreen_2[I,J] = k * im / 4 * ho1 * r_div_R_2[I,J]
end

@kernel function r_div_R!(r_1, r_2, @Const(R))
    I, J = @index(Global, NTuple)
    r_1[I,J] = r_1[I,J]/R[I,J]
    r_2[I,J] = r_2[I,J]/R[I,J]
end

@kernel function dot_normals!(dot, @Const(norm_11), @Const(norm_12), @Const(norm_21), @Const(norm_22))
    I, J = @index(Global, NTuple)
    dot[I,J] = norm_11[I] * norm_21[J] + norm_12[I] * norm_22[J]
end

@kernel function integrand_HyperSingular!(result_11, result_12, result_21, result_22, @Const(k), @Const(green), @Const(dot), @Const(womps_values_1), @Const(womps_values_2), @Const(womps_derivative_1), @Const(womps_derivative_2), @Const(wimps_values_1), @Const(wimps_values_2), @Const(wimps_derivative_1), @Const(wimps_derivative_2), @Const(j))
    I, J = @index(Global, NTuple)
    dot_k = dot[I,J] * k*k
    green_j = green[I,J] * j[I,J]
    result_11[I,J] = (womps_derivative_1[I] * wimps_derivative_1[J] - dot_k * womps_values_1[I] * wimps_values_1[J]) * green_j
    result_12[I,J] = (womps_derivative_1[I] * wimps_derivative_2[J] - dot_k * womps_values_1[I] * wimps_values_2[J]) * green_j
    result_21[I,J] = (womps_derivative_2[I] * wimps_derivative_1[J] - dot_k * womps_values_2[I] * wimps_values_1[J]) * green_j
    result_22[I,J] = (womps_derivative_2[I] * wimps_derivative_2[J] - dot_k * womps_values_2[I] * wimps_values_2[J]) * green_j
end

@kernel function sum_results!(a, result_11, result_12, result_21, result_22)
    #I, J = @index(Global, NTuple)
    K = @index(Group, Linear)
    L = @index(Local, Linear)
    @uniform groupsize = @groupsize()[1]
    @uniform threadIdxGlobal =  (K-1) * groupsize + L
    @print("\n @groupsize() = ", @groupsize()[1], " ", @groupsize()[2])
    @print("\n K = ",K," L = ",L, " groupsize = ", groupsize, " id? = ",threadIdxGlobal)

    @uniform  elements = prod(@ndrange())
    @uniform d = 1
    for d_ in 1:63 - leading_zeros(elements) +1# Loop over log2(n) stages of reduction
        @synchronize()
        for index in 1:2*d:elements - d # Step size 2d for each stage
            index_ = index * (threadIdxGlobal) + 1
            #@print("\n index = ",index_)
            if index_ <= elements - d
                result_11[index_] += result_11[index_ + d]
                result_12[index_] += result_12[index_ + d]
                result_21[index_] += result_21[index_ + d]
                result_22[index_] += result_22[index_ + d]
            end 
            #@print("\n index = ", index, " index + d = ", index + d)
        end
        @synchronize()
        d *= 2
    end

    a[1,1] = result_11[1]
    a[1,2] = result_12[1]
    a[2,1] = result_21[1]
    a[2,2] = result_22[1]
end
#r_temp, result_11, result_12, result_21, result_22, 
@kernel function combined_kernel!(r, r_temp,
    @Const(k), @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_cart_1), @Const(womps_cart_2), @Const(wimps_cart_1), @Const(wimps_cart_2), 
    @Const(womps_normals_1), @Const(womps_normals_2), @Const(wimps_normals_1), @Const(wimps_normals_2),
    @Const(womps_values_1), @Const(womps_values_2), @Const(womps_derivative_1), @Const(womps_derivative_2), 
    @Const(wimps_values_1), @Const(wimps_values_2), @Const(wimps_derivative_1), @Const(wimps_derivative_2))
    I, J = @index(Global, NTuple)

    j = womps_weights[I] * wimps_weights[J]
    womps_r = womps_cart_1[I] - wimps_cart_1[J]
    wimps_r = womps_cart_2[I] - wimps_cart_2[J]
    
    R = sqrt(womps_r^2 + wimps_r^2)
    kr = k * R
    hankels_order_0 = H_0_(kr)
    #@print("\n hankels_order_0 = ", real(hankels_order_0)," ", imag(hankels_order_0))
    #hankels_order_1 = H_1_(kr)
    #r_div_R_1 = womps_r / R
    #r_div_R_2 = wimps_r / R

    green = -im / 4 * hankels_order_0
    #gradgreen_1 = k * im / 4 * hankels_order_1 * r_div_R_1
    #gradgreen_2 = k * im / 4 * hankels_order_1 * r_div_R_2
    
    dot = womps_normals_1[I] * wimps_normals_1[J] + womps_normals_2[I] * wimps_normals_2[J]
    
    dot_k = dot * k*k
    green_j = green * j
    """
    result_11[I,J] = (womps_derivative_1[I] * wimps_derivative_1[J] - dot_k * womps_values_1[I] * wimps_values_1[J]) * green_j
    result_12[I,J] = (womps_derivative_1[I] * wimps_derivative_2[J] - dot_k * womps_values_1[I] * wimps_values_2[J]) * green_j
    result_21[I,J] = (womps_derivative_2[I] * wimps_derivative_1[J] - dot_k * womps_values_2[I] * wimps_values_1[J]) * green_j
    result_22[I,J] = (womps_derivative_2[I] * wimps_derivative_2[J] - dot_k * womps_values_2[I] * wimps_values_2[J]) * green_j
    """
    K = @index(Group, Linear)
    tid  = @index(Local, Linear)
    @uniform groupsize = @ndrange()[1]
    GlobalIdxMemory =  (K-1) * groupsize + tid 
    @print(@ndrange)
    T11 = @localmem ComplexF32 (groupsize)
    T12 = @localmem ComplexF32 (groupsize)
    T21 = @localmem ComplexF32 (groupsize)
    T22 = @localmem ComplexF32 (groupsize)
    @print("\n groupsize = ",groupsize)

    T11[tid] = (womps_derivative_1[I] * wimps_derivative_1[J] - dot_k * womps_values_1[I] * wimps_values_1[J]) * green_j
    T12[tid] = (womps_derivative_1[I] * wimps_derivative_2[J] - dot_k * womps_values_1[I] * wimps_values_2[J]) * green_j
    T21[tid] = (womps_derivative_2[I] * wimps_derivative_1[J] - dot_k * womps_values_2[I] * wimps_values_1[J]) * green_j
    T22[tid] = (womps_derivative_2[I] * wimps_derivative_2[J] - dot_k * womps_values_2[I] * wimps_values_2[J]) * green_j
    
    @synchronize()

    for d_ in ((63 - leading_zeros(groupsize))):-1:1# Loop over log2(n) stages of reduction
        d = 2 ^ (d_-1)
        @print("\n d = ",d)
        if tid + d <= groupsize
            T11[tid] += T11[tid + d]
            T12[tid] += T12[tid + d]
            T21[tid] += T21[tid + d]
            T22[tid] += T22[tid + d]
            #T[tid] += T[tid + d]
        end
        @synchronize()
    end 

    if (tid==1) 
        Atomix.@atomic r[1,1] += T11[1]
        Atomix.@atomic r[1,2] += T12[1]
        Atomix.@atomic r[2,1] += T21[1]
        Atomix.@atomic r[2,2] += T22[1]
    end


end

@kernel function combined_kernel_temp!(result_11, result_12, result_21, result_22,
    @Const(k), @Const(womps_weights), @Const(wimps_weights), 
    @Const(womps_cart_1), @Const(womps_cart_2), @Const(wimps_cart_1), @Const(wimps_cart_2), 
    @Const(womps_normals_1), @Const(womps_normals_2), @Const(wimps_normals_1), @Const(wimps_normals_2),
    @Const(womps_values_1), @Const(womps_values_2), @Const(womps_derivative_1), @Const(womps_derivative_2), 
    @Const(wimps_values_1), @Const(wimps_values_2), @Const(wimps_derivative_1), @Const(wimps_derivative_2))
    I, J = @index(Global, NTuple)

    j = womps_weights[I] * wimps_weights[J]
    womps_r = womps_cart_1[I] - wimps_cart_1[J]
    wimps_r = womps_cart_2[I] - wimps_cart_2[J]
    
    R = sqrt(womps_r^2 + wimps_r^2)
    kr = k * R
    hankels_order_0 = H_0_(kr)
    #@print("\n hankels_order_0 = ", real(hankels_order_0)," ", imag(hankels_order_0))
    #hankels_order_1 = H_1_(kr)
    #r_div_R_1 = womps_r / R
    #r_div_R_2 = wimps_r / R

    green = -im / 4 * hankels_order_0
    #gradgreen_1 = k * im / 4 * hankels_order_1 * r_div_R_1
    #gradgreen_2 = k * im / 4 * hankels_order_1 * r_div_R_2
    
    dot = womps_normals_1[I] * wimps_normals_1[J] + womps_normals_2[I] * wimps_normals_2[J]
    
    dot_k = dot * k*k
    green_j = green * j
    
    result_11[I,J] = (womps_derivative_1[I] * wimps_derivative_1[J] - dot_k * womps_values_1[I] * wimps_values_1[J]) * green_j
    result_12[I,J] = (womps_derivative_1[I] * wimps_derivative_2[J] - dot_k * womps_values_1[I] * wimps_values_2[J]) * green_j
    result_21[I,J] = (womps_derivative_2[I] * wimps_derivative_1[J] - dot_k * womps_values_2[I] * wimps_values_1[J]) * green_j
    result_22[I,J] = (womps_derivative_2[I] * wimps_derivative_2[J] - dot_k * womps_values_2[I] * wimps_values_2[J]) * green_j
end  

function momintegrals_gpu!(biop, tshs, bshs, tcell, bcell, z, strat::DoubleQuadRule)

    igd = Integrand(biop, tshs, bshs, tcell, bcell)

    womps = strat.outer_quad_points
    wimps = strat.inner_quad_points

    include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

    data = womps
    @show data
    #WEIGHT
    womps_weights = CuArray{Float32}(([entry.weight for entry in data]))

    #VALUES
    womps_values_1 =  CuArray{Float32}(([entry.value[1].value for entry in data]))
    womps_values_2 =  CuArray{Float32}(([entry.value[2].value for entry in data]))
    womps_derivative_1 =  CuArray{Float32}(([entry.value[1].derivative for entry in data]))
    womps_derivative_2 =  CuArray{Float32}(([entry.value[2].derivative for entry in data]))


    #POINT
    #womps_bary =  CuArray{Float32}(([entry.point.bary[1] for entry in data]))
    womps_cart_1 =   CuArray{Float32}(([entry.point.cart[1] for entry in data]))
    womps_cart_2 =   CuArray{Float32}(([entry.point.cart[2] for entry in data]))

    #   patch
    #womps_vertices_1 =  CuArray{Float32}(([entry.point.patch.vertices[1][1] for entry in data]))
    #womps_vertices_2 =  CuArray{Float32}(([entry.point.patch.vertices[1][2] for entry in data]))
    #womps_vertices_3 =  CuArray{Float32}(([entry.point.patch.vertices[2][1] for entry in data]))
    #womps_vertices_4 =  CuArray{Float32}(([entry.point.patch.vertices[2][2] for entry in data]))

    #womps_tangents_1 =  CuArray{Float32}(([entry.point.patch.tangents[1][1] for entry in data]))
    #womps_tangents_2 =  CuArray{Float32}(([entry.point.patch.tangents[1][2] for entry in data]))

    womps_normals_1 =  CuArray{Float32}(([entry.point.patch.normals[1][1] for entry in data]))
    womps_normals_2 =  CuArray{Float32}(([entry.point.patch.normals[1][2] for entry in data]))


    #womps_volume =  CuArray{Float32}(([entry.point.patch.volume for entry in data]))


    data = wimps
    #WEIGHT
    wimps_weights = CuArray{Float32}(([entry.weight for entry in data]))

    #VALUES
    wimps_values_1 =  CuArray{Float32}(([entry.value[1].value for entry in data]))
    wimps_values_2 =  CuArray{Float32}(([entry.value[2].value for entry in data]))
    wimps_derivative_1 =  CuArray{Float32}(([entry.value[1].derivative for entry in data]))
    wimps_derivative_2 =  CuArray{Float32}(([entry.value[2].derivative for entry in data]))


    #POINT
    #wimps_bary =  CuArray{Float32}(([entry.point.bary[1] for entry in data]))
    wimps_cart_1 =   CuArray{Float32}(([entry.point.cart[1] for entry in data]))
    wimps_cart_2 =   CuArray{Float32}(([entry.point.cart[2] for entry in data]))
    #wimps_cart = CuArray(Array([[e for e in entry.point.cart] for entry in data]))
    #   patch
    #wimps_vertices_1 =  CuArray{Float32}(([entry.point.patch.vertices[1][1] for entry in data]))
    #wimps_vertices_2 =  CuArray{Float32}(([entry.point.patch.vertices[1][2] for entry in data]))
    #wimps_vertices_3 =  CuArray{Float32}(([entry.point.patch.vertices[2][1] for entry in data]))
    #wimps_vertices_4 =  CuArray{Float32}(([entry.point.patch.vertices[2][2] for entry in data]))

    #wimps_tangents_1 =  CuArray{Float32}(([entry.point.patch.tangents[1][1] for entry in data]))
    #wimps_tangents_2 =  CuArray{Float32}(([entry.point.patch.tangents[1][2] for entry in data]))

    wimps_normals_1 =  CuArray{Float32}(([entry.point.patch.normals[1][1] for entry in data]))
    wimps_normals_2 =  CuArray{Float32}(([entry.point.patch.normals[1][2] for entry in data]))

    #wimps_volume =  CuArray{Float32}(([entry.point.patch.volume for entry in data]))

    len_womps = length(womps_cart_1)
    len_wimps = length(wimps_cart_1)
    
    if true
        result_11 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
        result_12 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
        result_21 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
        result_22 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)

        r = zeros(Complex{Float64}, 2, 2)# KernelAbstractions.zeros(backend, ComplexF32, 4, 4)
        r_temp = KernelAbstractions.zeros(backend, ComplexF32, 4, 4, 16)

        kernel! = combined_kernel_temp!(backend, 32, size(result_11))
        #r_temp, result_11, result_12, result_21, result_22,
        kernel!(result_11, result_12, result_21, result_22,
                biop.wavenumber, womps_weights, wimps_weights, 
                womps_cart_1, womps_cart_2, wimps_cart_1, wimps_cart_2, 
                womps_normals_1, womps_normals_2, wimps_normals_1, wimps_normals_2,
                womps_values_1, womps_values_2, womps_derivative_1, womps_derivative_2, 
                wimps_values_1, wimps_values_2, wimps_derivative_1, wimps_derivative_2, ndrange = (len_womps,len_wimps))

                

        

        KernelAbstractions.synchronize(backend)
        
        if backend == CUDABackend()
            r[1,1] = CUDA.@sync sum(result_11)
            r[1,2] = CUDA.@sync sum(result_12)
            r[2,1] = CUDA.@sync sum(result_21)
            r[2,2] = CUDA.@sync sum(result_22)
        else
            throw("implement backend")
        end 
        
        @show r
        return r
    end
    #@show length(wimps_values_1),length(wimps_values_2), length(womps_values_1),length(womps_values_2)


    j = KernelAbstractions.zeros(backend, Float32, len_womps, len_wimps)
    #groupsize = KernelAbstractions.isgpu(backend.backend_) ? 256 : 1024

    kernel! = calc_j!(backend, 32, size(j))
    kernel!(j, womps_weights, wimps_weights, ndrange = (len_womps, len_wimps))
    

    #kernel(KernelAbstractions.zeros(backend, Float32,  128, 128), KernelAbstractions.zeros(backend, Float32,  128),KernelAbstractions.zeros(backend, Float32,  128), ndrange = size(CUDA.fill(1.0f0, 128)))
    KernelAbstractions.synchronize(backend)



    r_1 = KernelAbstractions.zeros(backend, Float32, len_womps, len_wimps)
    kernel! = calc_r!(backend, 32, size(r_1))
    kernel!(r_1,womps_cart_1, wimps_cart_1, ndrange = (len_womps,len_wimps))

    println("")
    @show r_1

    r_2 = KernelAbstractions.zeros(backend, Float32, len_womps, len_wimps)
    kernel_2! = calc_r!(backend, 32, size(r_2))
    kernel_2!(r_2,womps_cart_2, wimps_cart_2, ndrange = (len_womps,len_wimps))
    
    KernelAbstractions.synchronize(backend)

    @show r_2

    
    #R= norm(r)
    R = KernelAbstractions.zeros(backend, Float32, len_womps, len_wimps)
    kernel! = calc_norm_2!(backend, 32, size(R))
    kernel!(R,r_1, r_2, ndrange = (len_womps,len_wimps))
    KernelAbstractions.synchronize(backend)

    
    k = biop.wavenumber
    kr = KernelAbstractions.zeros(backend, Float32, len_womps, len_wimps)
    kernel! = const_mul_vect!(backend, 32, size(kr))
    kernel!(kr,k, R, ndrange = length(kr))
    KernelAbstractions.synchronize(backend)

    

    hankels_order_0 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
    hankels_order_1 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
    
    kernel! = Hankel_type_2!(backend, 32, size(kr))
    kernel!(hankels_order_0, hankels_order_1, kr, ndrange = length(kr))

    
    KernelAbstractions.synchronize(backend)
    #@show hankels_order_0
    
    kernel! = r_div_R!(backend, 32, size(kr))
    kernel!(r_1, r_2, R, ndrange = length(kr))
    
    
    KernelAbstractions.synchronize(backend)


    green = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
    gradgreen_1 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
    gradgreen_2 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
    kernel! = green!(backend, 32, size(kr))
    kernel!(green, gradgreen_1, gradgreen_2, hankels_order_0, hankels_order_1, r_1, r_2, biop.wavenumber, ndrange = length(kr))

    KernelAbstractions.synchronize(backend)


    dot = KernelAbstractions.zeros(backend, Float32, len_womps, len_wimps)
    kernel! = dot_normals!(backend, 32, size(dot))
    kernel!(dot, womps_normals_1, womps_normals_2, wimps_normals_1, wimps_normals_2, ndrange = (len_womps,len_wimps))

    KernelAbstractions.synchronize(backend)

    result_11 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
    result_12 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
    result_21 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)
    result_22 = KernelAbstractions.zeros(backend, ComplexF32, len_womps, len_wimps)


    #@show size(result_11), size(green), size(dot), size(womps_values_1), size(womps_derivative_1), size(wimps_values_1), size(wimps_derivative_1), size(j)
    kernel! = integrand_HyperSingular!(backend, 32, size(result_11))
    kernel!(result_11, result_12, result_21, result_22, biop.wavenumber, green, dot, womps_values_1, womps_values_2, womps_derivative_1, womps_derivative_2, wimps_values_1, wimps_values_2, wimps_derivative_1, wimps_derivative_2, j, ndrange = (len_womps,len_wimps))

    KernelAbstractions.synchronize(backend)

    r = zeros(ComplexF32, 4, 4)
    if backend == CUDABackend()
        r[1,1] = CUDA.@sync sum(result_11)
        r[1,2] = CUDA.@sync sum(result_12)
        r[2,1] = CUDA.@sync sum(result_21)
        r[2,2] = CUDA.@sync sum(result_22)
    else
        throw("implement backend")
    end 
    #@show r
    return r
end
