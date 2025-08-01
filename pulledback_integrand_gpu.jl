using KernelAbstractions
using KernelAbstractions: @atomic
using KernelAbstractions.Extras: @unroll

@kernel function test_toll_CommonFace_this_one!(r1, r2, @Const(t), @Const(s), @Const(tol_sq))
    I = @index(Global)
    
    if sqrt((t[I] - s[1])^2 + (t[I + 3] - s[4])^2 + (t[I + 6] - s[7])^2) < tol_sq
        r2[I] = 1
    end
    if sqrt((t[I] - s[2])^2 + (t[I + 3] - s[5])^2 + (t[I + 6] - s[8])^2) < tol_sq
        r2[I] = 2
    end
    if sqrt((t[I] - s[3])^2 + (t[I + 3] - s[6])^2 + (t[I + 6] - s[9])^2) < tol_sq
        r2[I] = 3
    end
    r1[I] = I
end


const tol_Float64 = 1e3 * eps(Float64)
@kernel function test_toll!(vertices1, vertices2, tangents1, tangents2, @Const(vertices1_), @Const(vertices2_), @Const(indexes), @Const(tcell_vertices), @Const(bcell_vertices), T::CommonFaceCustomGpuData)
    groupsize = prod(@groupsize())
    
    I = @localmem Int (512, 3)
    J = @localmem Int (512, 3)
    
    global_index = @index(Global)
    local_index = @index(Local)
    
    @unroll for unroll in 1:3
        I[local_index, unroll] = 0 
        J[local_index, unroll] = 0 
    end

    @synchronize

    p = indexes[global_index, 1]
    q = indexes[global_index, 2]

    @unroll for i in 1:3
        @unroll for j in 1:3
            R = 0
            @unroll for unroll in 1:3
                R += (tcell_vertices[p, unroll, i] - bcell_vertices[q, unroll, j])^2
            end
            if sqrt(R) < tol_Float64
                J[local_index, i] = j
            end
        end
    end
    @unroll for unroll in 1:3
        I[local_index, unroll] = unroll
    end
    

    pulledback_integrand_gpu!(tangents1, tangents2, vertices1, vertices2, vertices1_, vertices2_, I, J, local_index, global_index)
end

@kernel function test_toll!(vertices1, vertices2, tangents1, tangents2, @Const(vertices1_), @Const(vertices2_), @Const(indexes), @Const(tcell_vertices), @Const(bcell_vertices), T::CommonEdgeCustomGpuData)
    groupsize = prod(@groupsize())
    
    I = @localmem Int (512, 3)
    J = @localmem Int (512, 3)
    
    global_index = @index(Global)
    local_index = @index(Local)
    
    @unroll for unroll in 1:3
        I[local_index, unroll] = 0 
        J[local_index, unroll] = 0 
    end

    @synchronize

    p = indexes[global_index, 1]
    q = indexes[global_index, 2]


    e = 1
    @unroll for i in 1:3
        @unroll for j in 1:3
            R = 0
            @unroll for unroll in 1:3
                R += (tcell_vertices[p, unroll, i] - bcell_vertices[q, unroll, j])^2
            end
            if sqrt(R) < tol_Float64
                J[local_index, e] = e <= 3 ? j : J[local_index, e]
                I[local_index, e] = e <= 3 ? i : I[local_index, e]
                e += e <= 3 ? 1 : 0
            end
        end
    end

    setdiff_lengthe_3(J, local_index, e == 1, e == 2)
    setdiff_lengthe_3(I, local_index, e == 1, e == 2)

    circshift_length_3(J, local_index)
    circshift_length_3(I, local_index)

    pulledback_integrand_gpu!(tangents1, tangents2, vertices1, vertices2, vertices1_, vertices2_, I, J, local_index, global_index)
end

function circshift_length_3(array, index)
    temp = array[index, 1]
    array[index, 1] = array[index, 2]
    array[index, 2] = array[index, 3]
    array[index, 3] = temp
end

function setdiff_lengthe_3(array, local_index, e_is_1, e_is_2)
    array[local_index, 1] = e_is_1 ? 1 : array[local_index, 1]
    array[local_index, 2] = e_is_1 ? 2 : (e_is_2 ? (array[local_index, 1] == 1 ? 2 : 1) : array[local_index, 2]) 
    array[local_index, 3] = 6 - array[local_index, 1] - array[local_index, 2]
end

@kernel function test_toll!(vertices1, vertices2, tangents1, tangents2, @Const(vertices1_), @Const(vertices2_), @Const(indexes), @Const(tcell_vertices), @Const(bcell_vertices), T::CommonVertexCustomGpuData)
    groupsize = prod(@groupsize())
    
    I = @localmem Int (512, 3)
    J = @localmem Int (512, 3)
    
    global_index = @index(Global)
    local_index = @index(Local)
    
    @unroll for unroll in 1:3
        I[local_index, unroll] = 0 
        J[local_index, unroll] = 0 
    end

    @synchronize

    p = indexes[global_index, 1]
    q = indexes[global_index, 2]

    e = 1
    @unroll for i in 1:3
        @unroll for j in 1:3
            R = 0
            @unroll for unroll in 1:3
                R += (tcell_vertices[p, unroll, i] - bcell_vertices[q, unroll, j])^2
            end
            if sqrt(R) < tol_Float64
                J[local_index, e] = e <= 2 ? j : J[local_index, e]
                I[local_index, e] = e <= 2 ? i : I[local_index, e]
                e += e <= 2 ? 1 : 0
            end
        end
    end

    setdiff_lengthe_3(J, local_index, e == 1, e == 2)
    setdiff_lengthe_3(I, local_index, e == 1, e == 2)

    pulledback_integrand_gpu!(tangents1, tangents2, vertices1, vertices2, vertices1_, vertices2_, I, J, local_index, global_index)
end


@kernel function test_toll_CommonVertex!(r1, r2, @Const(t), @Const(s), @Const(tol_sq))
    I = @index(Global)
    shared_r1 = @localmem Int (9)
    shared_r2 = @localmem Int (9)
    for i in 1:9
        shared_r1[i] = 0
        shared_r2[i] = 0
    end
    @synchronize
    if     sqrt((t[1, I] - s[1, 1])^2 + (t[2, I] - s[2, 1])^2 + (t[3, I] - s[3, 1])^2) < tol_sq
        shared_r1[(I-1)*3+1] = I
        shared_r2[(I-1)*3+1] = 1
    elseif sqrt((t[1, I] - s[1, 2])^2 + (t[2, I] - s[2, 2])^2 + (t[3, I] - s[3, 2])^2) < tol_sq
        shared_r1[(I-1)*3+2] = I
        shared_r2[(I-1)*3+2] = 2
    elseif sqrt((t[1, I] - s[1, 3])^2 + (t[2, I] - s[2, 3])^2 + (t[3, I] - s[3, 3])^2) < tol_sq
        shared_r1[(I-1)*3+3] = I
        shared_r2[(I-1)*3+3] = 3
    end
    @synchronize
    if I == 1
        for i in 1:9
            if shared_r1[i] != 0
                r1[1] = shared_r1[i]
                if r1[1] == 1
                    r1[2] = 2
                else
                    r1[2] = 1
                end
                r1[3] = 6 - r1[1] - r1[2]
                break
            end
        end
    elseif I == 2
        for i in 1:9
            if shared_r2[i] != 0
                r2[1] = shared_r2[i]
                if r2[1] == 1
                    r2[2] = 2
                else
                    r2[2] = 1
                end
                r2[3] = 6 - r2[1] - r2[2]
                break
            end
        end
    end
end


function pulledback_integrand_gpu!(tangents1, tangents2, vertices1, vertices2, vertices1_, vertices2_, I, J, local_index, global_index)
    @unroll for unroll in 1:3
        vertices1[global_index, unroll, 1] = vertices1_[I[local_index, unroll], 1]
        vertices1[global_index, unroll, 2] = vertices1_[I[local_index, unroll], 2]
    end

    @unroll for unroll in 1:2
        tangents1[global_index, unroll, 1] = vertices1[global_index, unroll, 1] - vertices1[global_index, 3, 1]
        tangents1[global_index, unroll, 2] = vertices1[global_index, unroll, 2] - vertices1[global_index, 3, 2]
    end

    @unroll for unroll in 1:3
        vertices2[global_index, unroll, 1] = vertices2_[J[local_index, unroll], 1]
        vertices2[global_index, unroll, 2] = vertices2_[J[local_index, unroll], 2]
    end

    @unroll for unroll in 1:2
        tangents2[global_index, unroll, 1] = vertices2[global_index, unroll, 1] - vertices2[global_index, 3, 1]
        tangents2[global_index, unroll, 2] = vertices2[global_index, unroll, 2] - vertices2[global_index, 3, 2]
    end
end
