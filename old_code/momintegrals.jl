

module Zzz
    export momintegrals_gpu_this!
    include("strategy_sauterschwabints.jl") 
    using .Strategies             

    using KernelAbstractions, CUDA#, Atomix
    using KernelAbstractions: @atomic, @atomicswap, @atomicreplace
    using BEAST, CompScienceMeshes
    using StaticArrays
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

    @kernel function test_toll_CommonFace_this_one_optimized!(r, @Const(t), @Const(s), @Const(tol_sq))
        I = @index(Global)
        shared_s = @localmem Int (9)
        shared_s = s[1:9]
        
        if     sqrt((t[I] - shared_s[1])^2 + (t[I + 3] - shared_s[4])^2 + (t[I + 6] - shared_s[7])^2) < tol_sq
            r[I] = 1
        elseif sqrt((t[I] - shared_s[2])^2 + (t[I + 3] - shared_s[5])^2 + (t[I + 6] - shared_s[8])^2) < tol_sq
            r[I] = 2
        elseif sqrt((t[I] - shared_s[3])^2 + (t[I + 3] - shared_s[6])^2 + (t[I + 6] - shared_s[9])^2) < tol_sq
            r[I] = 3
        end
    end

    @kernel function test_toll_CommonEdge!(r1, r2, @Const(t), @Const(s), @Const(tol_sq))
        I = @index(Global)
        if     sqrt((t[I] - s[1])^2 + (t[I + 3] - s[4])^2 + (t[I + 6] - s[7])^2) < tol_sq
            r1[I] = I
            r2[I] = 1
        elseif sqrt((t[I] - s[2])^2 + (t[I + 3] - s[5])^2 + (t[I + 6] - s[8])^2) < tol_sq
            r1[I] = I
            r2[I] = 2
        elseif sqrt((t[I] - s[3])^2 + (t[I + 3] - s[6])^2 + (t[I + 6] - s[9])^2) < tol_sq
            r1[I] = I
            r2[I] = 3
        end
        @synchronize
        if I == 1
            for val in 1:3
                if val != r1[1] && val != r1[2]
                    r1[3] = val
                    break
                end
            end
            temp = r1[1]
            for idx in 1:2
                r1[idx] = r1[idx + 1]
            end
            r1[3] = temp
        elseif I == 2
            for val in 1:3
                if val != r2[1] && val != r2[2]
                    r2[3] = val
                    break
                end
            end
            temp = r2[1]
            for idx in 1:2
                r2[idx] = r2[idx + 1]
            end
            r2[3] = temp
        end
    end

    @kernel function test_toll_CommonEdge_optimized!(r1, r2, @Const(t), @Const(s), @Const(tol_sq))
        I = @index(Global)
        shared_r1 = @localmem Int (9)
        shared_r2 = @localmem Int (9)
        for i in 1:9
            shared_r1[i] = 0
            shared_r2[i] = 0
        end
        @synchronize
        # if I == 1
        #     @print("\n t[1,1] = ", t[1,1], " t[1,2] = ", t[1,2], " t[1,3] = ", t[1,3], " t[2,1] = ", t[2,1], " t[2,2] = ", t[2,2], " t[2,3] = ", t[2,3], " t[3,1] = ", t[3,1], " t[3,2] = ", t[3,2], " t[3,3] = ", t[3,3])
        #     @print("\n t[1,1] = ", s[1,1], " t[1,2] = ", s[1,2], " t[1,3] = ", s[1,3], " t[2,1] = ", s[2,1], " t[2,2] = ", s[2,2], " t[2,3] = ", s[2,3], " t[3,1] = ", s[3,1], " t[3,2] = ", s[3,2], " t[3,3] = ", s[3,3])
        # end

        # @print("\n I = ", I,
        #         " \nt[1, I] = ", t[1, I], " t[2, I] = ", t[2, I], " t[3, I] = ", t[3, I], 
        #         " \ns[1, 1] = ", s[1, 1], " s[2, 1] = ", s[2, 1], " s[3, 1] = ", s[3, 1], 
        #         " \ns[1, 2] = ", s[1, 2], " s[2, 2] = ", s[2, 2], " s[3, 2] = ", s[3, 2], 
        #         " \ns[1, 3] = ", s[1, 3], " s[2, 3] = ", s[2, 3], " s[3, 3] = ", s[3, 3])

        if     (t[1, I] - s[1, 1])^2 + (t[2, I] - s[2, 1])^2 + (t[3, I] - s[3, 1])^2 < tol_sq
            shared_r1[(I-1)*3+1] = I
            shared_r2[(I-1)*3+1] = 1
        elseif (t[1, I] - s[1, 2])^2 + (t[2, I] - s[2, 2])^2 + (t[3, I] - s[3, 2])^2 < tol_sq
            shared_r1[(I-1)*3+2] = I
            shared_r2[(I-1)*3+2] = 2
        elseif (t[1, I] - s[1, 3])^2 + (t[2, I] - s[2, 3])^2 + (t[3, I] - s[3, 3])^2 < tol_sq
            shared_r1[(I-1)*3+3] = I
            shared_r2[(I-1)*3+3] = 3
        end
        @synchronize
        if I == 1
            t = 1
            for i in 1:9
                if shared_r1[i] != 0
                    if t == 1
                        r1[3] = shared_r1[i]
                    elseif t == 2
                        r1[1] = shared_r1[i]
                        r1[2] = 6 - r1[1] - r1[3]
                        break
                    end
                    t += 1
                end
            end
        elseif I == 2
            t = 1
            for i in 1:9
                if shared_r2[i] != 0
                    if t == 1
                        r2[3] = shared_r2[i]
                    elseif t == 2
                        r2[1] = shared_r2[i]
                        r2[2] = 6 - r2[1] - r2[3]
                        break
                    end
                    t += 1
                end
            end
        end
    end

    #needs only 2 threads instead of 3 per 
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




    @kernel function pulledback_integrand_gpu!(tangents1, tangents2, vertices1, vertices2, @Const(I), @Const(J))
        i = @index(Global)
        vertices_temp = @localmem Int (3,2) #make private
        vertices_temp[1,1] = 0
        vertices_temp[1,2] = 0
        vertices_temp[2,1] = 0
        vertices_temp[2,2] = 0
        vertices_temp[3,1] = 0
        vertices_temp[3,2] = 0
        @synchronize

        # ichart1[1] = vd[I[1]][0]                      # ichart2[1] = vd[J[1]][0]
        # ichart1[2] = vd[I[1]][1]                      # ichart2[2] = vd[J[1]][1]
        # ichart1[3] = vd[I[2]][0]                      # ichart2[3] = vd[J[2]][0]
        # ichart1[4] = vd[I[2]][1]                      # ichart2[4] = vd[J[2]][1]
        # ichart1[5] = vd[I[3]][0]                      # ichart2[5] = vd[J[3]][0]
        # ichart1[6] = vd[I[3]][1]                      # ichart2[6] = vd[J[3]][1]

        # tangents1[1] = ichart1[1] - ichart1[5]        # tangents2[1] = ichart1[1] - ichart1[5]
        # tangents1[2] = ichart1[2] - ichart1[6]        # tangents2[2] = ichart1[2] - ichart1[6]
        # tangents1[3] = ichart1[3] - ichart1[5]        # tangents2[3] = ichart1[3] - ichart1[5]
        # tangents1[4] = ichart1[4] - ichart1[6]        # tangents2[4] = ichart1[4] - ichart1[6]

        vertices_temp[1, 1] = vertices1[I[1],1]
        vertices_temp[1, 2] = vertices1[I[1],2]
        vertices_temp[2, 1] = vertices1[I[2],1]
        vertices_temp[2, 2] = vertices1[I[2],2]
        vertices_temp[3, 1] = vertices1[I[3],1]
        vertices_temp[3, 2] = vertices1[I[3],2]

        vertices1[1, 1] = vertices_temp[1, 1]
        vertices1[1, 2] = vertices_temp[1, 2]
        vertices1[2, 1] = vertices_temp[2, 1]
        vertices1[2, 2] = vertices_temp[2, 2]
        vertices1[3, 1] = vertices_temp[3, 1]
        vertices1[3, 2] = vertices_temp[3, 2]

        tangents1[1, 1] = vertices_temp[1, 1] - vertices_temp[3, 1]
        tangents1[1, 2] = vertices_temp[1, 2] - vertices_temp[3, 2]
        tangents1[2, 1] = vertices_temp[2, 1] - vertices_temp[3, 1]
        tangents1[2, 2] = vertices_temp[2, 2] - vertices_temp[3, 2]




        vertices_temp[1, 1] = vertices2[J[1],1]
        vertices_temp[1, 2] = vertices2[J[1],2]
        vertices_temp[2, 1] = vertices2[J[2],1]
        vertices_temp[2, 2] = vertices2[J[2],2]
        vertices_temp[3, 1] = vertices2[J[3],1]
        vertices_temp[3, 2] = vertices2[J[3],2]
        
        vertices2[1, 1] = vertices_temp[1, 1]
        vertices2[1, 2] = vertices_temp[1, 2]
        vertices2[2, 1] = vertices_temp[2, 1]
        vertices2[2, 2] = vertices_temp[2, 2]
        vertices2[3, 1] = vertices_temp[3, 1]
        vertices2[3, 2] = vertices_temp[3, 2]

        tangents2[1, 1] = vertices2[1, 1] - vertices2[3, 1]
        tangents2[1, 2] = vertices2[1, 2] - vertices2[3, 2]
        tangents2[2, 1] = vertices2[2, 1] - vertices2[3, 1]
        tangents2[2, 2] = vertices2[2, 2] - vertices2[3, 2]

        # volume1[1] = (tangents1[1] * tangents1[4] - tangents1[2] * tangents1[3])/2
        # volume2[1] = (tangents2[1] * tangents2[4] - tangents2[2] * tangents2[3])/2
    end

    function load(qps, i)
        i_ = i - 1
        index_1 = div(i_, 64) + 1
        η1 = qps[index_1, 1]
        w1 = qps[index_1, 2]
        index_2 = div(rem(i_, 64), 16) + 1
        η2 = qps[index_2, 1]
        w2 = qps[index_2, 2]
        index_3 = div(rem(i_, 16), 4) + 1
        η3 = qps[index_3, 1]
        w3 = qps[index_3, 2]
        index_4 = rem(i_, 4) + 1
        ξ =  qps[index_4, 1]
        w4 = qps[index_4, 2]
        return η1, w1, η2, w2, η3, w3, ξ, w4
    end

    function multiply_all(igd_Integrands, i, j, k, l,  w1, w2, w3, w4, ξ, η1, η2)
        for local_iter in 1:9
            # tss[i, local_iter] = value * (ξ^3) * ((η1)^2) * (η2) *  w1 * w2 * w3 * w4
            igd_Integrands[i, j, k, l, local_iter] *= w1 * w2 * w3 * w4 * (ξ^3) * ((η1)^2) * (η2)
        end
    end

    function multiply(igd_Integrands, i, j, k, l, ξ, η1, η2)
        for local_iter in 1:9
            igd_Integrands[i, j, k, l, local_iter] *= (ξ^3) * ((η1)^2) * (η2)
        end
    end

    function multiply_partial(igd_Integrands, i, j, k, l, w1, w2, w3, w4, ξ, η2)
        for local_iter in 1:9
            igd_Integrands[i, j, k, l, local_iter] *= w1 * w2 * w3 * w4 * (ξ^3) * η2
        end
    end

    function multiply_w(igd_Integrands, i, j, k, l, w1, w2, w3, w4)
        for local_iter in 1:9
            igd_Integrands[i, j, k, l, local_iter] *= w1 * w2 * w3 * w4
        end
    end

    function reduce(r, igd_Integrands, i)
        tot = 256
        ((63 - leading_zeros(256)))
        @unroll for iter in 8:-1:1
            d = 2 ^ (iter-1)
            if i <= d
                for local_iter in 1:9
                    igd_Integrands[i, local_iter] += igd_Integrands[i + d, local_iter]
                end
            end
            @synchronize()
        end

        if i == 1
            @unroll for local_iter in 1:9
                r[local_iter] = igd_Integrands[1,local_iter]
            end
        end
    end

    const inv_4pi = 1/(4pi)  
    function igd_Integrand_distilled_and_mul(i, j, k, l, igd_Integrands, u_, v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mul_)

        bary_x_1 = fc1v[3, 1] + fc1t[1, 1] * u_[1] + fc1t[2, 1] * u_[2]
        bary_x_2 = fc1v[3, 2] + fc1t[1, 2] * u_[1] + fc1t[2, 2] * u_[2]
        
        
        # @print("\n bary_x_1 = ", bary_x_1, " bary_x_2 = ", bary_x_2)

        cart_x_1 = itecv[1, 3] + itect[1, 1] * bary_x_1 + itect[1, 2] * bary_x_2
        cart_x_2 = itecv[2, 3] + itect[2, 1] * bary_x_1 + itect[2, 2] * bary_x_2
        cart_x_3 = itecv[3, 3] + itect[3, 1] * bary_x_1 + itect[3, 2] * bary_x_2
    
        bary_y_1 = fc2v[3, 1] + fc2t[1, 1] * v_[1] + fc2t[2, 1] * v_[2]
        bary_y_2 = fc2v[3, 2] + fc2t[1, 2] * v_[1] + fc2t[2, 2] * v_[2]
    
        cart_y_1 = itrcv[1, 3] + itrct[1, 1] * bary_y_1 + itrct[1, 2] * bary_y_2
        cart_y_2 = itrcv[2, 3] + itrct[2, 1] * bary_y_1 + itrct[2, 2] * bary_y_2
        cart_y_3 = itrcv[3, 3] + itrct[3, 1] * bary_y_1 + itrct[3, 2] * bary_y_2
    
        R = sqrt((cart_x_1 - cart_y_1)^2 + (cart_x_2 - cart_y_2)^2 + (cart_x_3 - cart_y_3)^2)
        green =  exp(-γ[1]*R) * inv_4pi / R
    
        f_1 = bary_x_1
        f_2 = bary_x_2
        f_3 = 1 - bary_x_1 - bary_x_2
        g_1 = bary_y_1
        g_2 = bary_y_2
        g_3 = 1 - bary_y_1 - bary_y_2
    
        jacobian_x_mul_jacobian_y_mul_green = 4 * itecvo[1] * itrcvo[1] * α[1] * green * mul_
        
        jacobian_x_mul_jacobian_y_mul_green_f_1 = jacobian_x_mul_jacobian_y_mul_green * f_1
        jacobian_x_mul_jacobian_y_mul_green_f_2 = jacobian_x_mul_jacobian_y_mul_green * f_2
        jacobian_x_mul_jacobian_y_mul_green_f_3 = jacobian_x_mul_jacobian_y_mul_green * f_3
    
        igd_Integrands[i, j, k, l, 1] += (jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)
        igd_Integrands[i, j, k, l, 2] += (jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)
        igd_Integrands[i, j, k, l, 3] += (jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)
    
        igd_Integrands[i, j, k, l, 4] += (jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)
        igd_Integrands[i, j, k, l, 5] += (jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)
        igd_Integrands[i, j, k, l, 6] += (jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    
        igd_Integrands[i, j, k, l, 7] += (jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)
        igd_Integrands[i, j, k, l, 8] += (jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)
        igd_Integrands[i, j, k, l, 9] += (jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)
    end

    @kernel function sauterschwab_parameterized_CommonFace_gpu!(r, tss_real, tss_imag, @Const(qps), @Const(test_vert), @Const(trail_vert), @Const(test_tan), @Const(trail_tan), @Const(test_vol), @Const(trail_vol), @Const(vertices1), @Const(vertices2), @Const(tangents1), @Const(tangents2), @Const(γ), @Const(α))
        i, j, k, l, m =  @index(Global, NTuple)
        i_, j_, k_, l_, m_ =  @index(Local, NTuple)
        if m_ > 1
            @print("\n  WARNING: racing conditions will occur")
        end
        Group_number = @index(Group, Linear)
        igd_Integrands = @localmem ComplexF64 (4,4,4,4, 9) # 4^4, 9

        @unroll for unroll in 1:9
            igd_Integrands[i_, j_, k_, l_,unroll] = 0
        end
        # @synchronize

        η1 = qps[i, 1]
        w1 = qps[i, 2]
        η2 = qps[j, 1]
        w2 = qps[j, 2]
        η3 = qps[k, 1]
        w3 = qps[k, 2]
        ξ =  qps[l, 1]
        w4 = qps[l, 2]

        mul_ = w1 * w2 * w3 * w4 * (ξ^3) * ((η1)^2) * (η2)
        # @print("\n i,j,k,l = ", i,j,k,l, " vertices1[3, 1] = ", vertices1[3, 1], " tangents1[1, 1] = ", tangents1[1, 1], " u_[1] = ", (1 - ξ), " tangents1[1, 2] = ", tangents1[1, 2], " u_[2] = ", (ξ - ξ * η1 + ξ * η1 * η2))
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_,  igd_Integrands,(1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), (1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 2
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), (1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 3
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), (1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 4
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), (1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 5
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), (1 - ξ, ξ * η1 * (1 - η2)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 6
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - ξ, ξ * η1 * (1 - η2)), (1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # end
        
        @unroll for unroll in 1:9
            @atomic tss_real[i, j, k, l, unroll] += real(igd_Integrands[i_, j_, k_, l_, unroll])
            @atomic tss_imag[i, j, k, l, unroll] += imag(igd_Integrands[i_, j_, k_, l_, unroll])
        end
        
        
        # reduce(r, igd_Integrands, i)
    end

    @kernel function sauterschwab_parameterized_CommonEdge_gpu!(r, tss, tss_real, tss_imag, @Const(qps), @Const(test_vert), @Const(trail_vert), @Const(test_tan), @Const(trail_tan), @Const(test_vol), @Const(trail_vol), @Const(vertices1), @Const(vertices2), @Const(tangents1), @Const(tangents2), @Const(γ), @Const(α))
        i, j, k, l, m =  @index(Global, NTuple)
        i_, j_, k_, l_, m_ =  @index(Local, NTuple)
        if m_ > 1
            @print("\n  WARNING: racing conditions will occur")
        end
        Group_number = @index(Group, Linear)
        igd_Integrands = @localmem ComplexF64 (4,4,4,4, 9) # 4^4, 9
        
        @unroll for unroll in 1:9
            igd_Integrands[i_, j_, k_, l_,unroll] = 0
        end
        # @synchronize

        η1 = qps[i, 1]
        w1 = qps[i, 2]
        η2 = qps[j, 1]
        w2 = qps[j, 2]
        η3 = qps[k, 1]
        w3 = qps[k, 2]
        ξ =  qps[l, 1]
        w4 = qps[l, 2]
        
        ξη1 = ξ * η1
        η1η2 = η1 * η2
        η2η3 = η2 * η3
        η1η2η3 = η1η2 * η3

        # if m == 1
        mul_=  (ξ^3) * ((η1)^2) * (η2) * w1 * w2 * w3 * w4
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - ξ, ξη1), (1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 2
        mul_=  (ξ^3) * ((η1)^2) * (η2) * w1 * w2 * w3 * w4
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), (1 - ξ, ξη1 * η2η3), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 3
        mul_=  (ξ^3) * ((η1)^2) * (η2) * w1 * w2 * w3 * w4
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), (1 - ξ, ξη1), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 4
        mul_=  (ξ^3) * ((η1)^2) * (η2) * w1 * w2 * w3 * w4
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - ξ * (1 - η1η2η3), ξη1 * (1 - η2η3)), (1 - ξ, ξη1 * η2), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 5
        mul_= (ξ^3) * ((η1)^2)* w1 * w2 * w3 * w4
        igd_Integrand_distilled_and_mul(i_, j_, k_, l_, igd_Integrands,(1 - ξ, ξη1 * η3), (1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # end 
        
        
        @unroll for unroll in 1:9
            @atomic tss_real[i, j, k, l, unroll] += real(igd_Integrands[i_, j_, k_, l_, unroll])
            @atomic tss_imag[i, j, k, l, unroll] += imag(igd_Integrands[i_, j_, k_, l_, unroll])
        end

        # @synchronize

        # reduce(r, igd_Integrands, i)
    end

    @kernel function sauterschwab_parameterized_CommonVertex_gpu!(r, tss_real, tss_imag, @Const(qps), @Const(test_vert), @Const(trail_vert), @Const(test_tan), @Const(trail_tan), @Const(test_vol), @Const(trail_vol), @Const(vertices1), @Const(vertices2), @Const(tangents1), @Const(tangents2), @Const(γ), @Const(α))
        i, j, k, l, m =  @index(Global, NTuple)
        i_, j_, k_, l_, m_ =  @index(Local, NTuple)
        if m_ > 1
            @print("\n  WARNING: racing conditions will occur")
        end
        Group_number = @index(Group, Linear)
        igd_Integrands = @localmem ComplexF64 (4,4,4,4, 9) # 4^4, 9
        
        @unroll for unroll in 1:9
            igd_Integrands[i_, j_, k_, l_,unroll] = 0
        end
        # @synchronize
        
        η1 = qps[i, 1]
        w1 = qps[i, 2]
        η2 = qps[j, 1]
        w2 = qps[j, 2]
        η3 = qps[k, 1]
        w3 = qps[k, 2]
        ξ =  qps[l, 1]
        w4 = qps[l, 2]

        ξη1 = ξ * η1
        ξη2 = ξ * η2
    
        # if m == 1
        mul_ = w1 * w2 * w3 * w4 * (ξ^3) * η2
        # if i * j * k * l == 1
        igd_Integrand_distilled_and_mul(i, j, k, l, igd_Integrands,(1 - ξ, ξη1), (1 - ξη2, ξη2 * η3), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # elseif m == 2
        igd_Integrand_distilled_and_mul(i, j, k, l, igd_Integrands,(1 - ξη2, ξη2 * η3), (1 - ξ, ξη1), vertices1, tangents1, test_vert, test_tan, vertices2, tangents2, trail_vert, trail_tan, test_vol, trail_vol, γ, α, mul_)
        # for unroll_offset in 1:9 
        #     @print("\n ",real(igd_Integrands[1,1,1,1,unroll_offset]), " ", imag(igd_Integrands[1,1,1,1,unroll_offset]), "im")
        # end
        # end

        
        @unroll for unroll in 1:9
            @atomic tss_real[i, j, k, l, unroll] += real(igd_Integrands[i_, j_, k_, l_, unroll])
            @atomic tss_imag[i, j, k, l, unroll] += imag(igd_Integrands[i_, j_, k_, l_, unroll])
        end

        # reduce(r, igd_Integrands, i)
    end
    function momintegrals_gpu_this!(op,
        test_local_space, trial_local_space,
        test_chart, trial_chart,
        out, rule)
        include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

        t_1 = time()
        num_tshapes = numfunctions(test_local_space, domain(test_chart))
        num_bshapes = numfunctions(trial_local_space, domain(trial_chart))

        test_vert = CuArray{Float64}(hcat(map(x -> collect(x), test_chart.vertices)...))
        trail_vert = CuArray{Float64}(hcat(map(x -> collect(x), trial_chart.vertices)...)) 
        test_tan = CuArray{Float64}(hcat(map(x -> collect(x), test_chart.tangents)...))  
        trail_tan  = CuArray{Float64}(hcat(map(x -> collect(x), trial_chart.tangents)...))  
        test_vol = CuArray{Float64}(hcat(map(x -> collect(x), test_chart.volume)...))  
        trail_vol = CuArray{Float64}(hcat(map(x -> collect(x), trial_chart.volume)...))  
        qps = CuArray{Float64}(hcat(map(x -> collect(x), rule.qps)...)')
        γ = CuArray{ComplexF64}(hcat(map(x -> collect(x), op.gamma)...))
        α = CuArray{ComplexF64}(hcat(map(x -> collect(x), op.alpha)...))

        # @show size(test_vert)
        # @show size(trail_vert)
        # @show size(test_tan)
        # @show size(trail_tan)
        # @show size(test_vol)
        # @show size(trail_vol)
        # @show size(qps)
        # throw("jsdqklfm")

        t_2 = time()

        tol_sq = 1e6 * eps(eltype(test_chart.vertices[1])) ^ 2
        I = KernelAbstractions.zeros(backend, Int64,3)
        J = KernelAbstractions.zeros(backend, Int64,3)
        
        if isa(rule, BEAST.SauterSchwabQuadrature.CommonFace)
            kernel! = test_toll_CommonFace_this_one!(backend)
            # println("CommonFace")
        elseif isa(rule, BEAST.SauterSchwabQuadrature.CommonEdge)
            kernel! = test_toll_CommonEdge_optimized!(backend)
            # println("CommonEdge")
        elseif isa(rule, BEAST.SauterSchwabQuadrature.CommonVertex)
            kernel! = test_toll_CommonVertex!(backend)
            # println("CommonVertex")
        else
            println("other") #mss PositiveDistance
        end
        kernel!(I, J, test_vert, trail_vert, tol_sq, ndrange = length(I))
        KernelAbstractions.synchronize(backend)
        # @show I, J
        t_3 = time()

        num_tshapes = 3
        num_bshapes = 3

        tangents1 = KernelAbstractions.zeros(backend, Float64, (2, 2))
        tangents2 = KernelAbstractions.zeros(backend, Float64, (2, 2))
        vertices1 = CuArray{Float64}([1.0 0.0; 0.0 1.0; 0.0 0.0])
        vertices2 = CuArray{Float64}([1.0 0.0; 0.0 1.0; 0.0 0.0])

        @show size(test_vert), size(trail_vert)
        @show size(test_tan), size(trail_tan)
        @show size(test_vol), size(trail_vol)
        @show size(vertices1), size(vertices2)
        @show size(tangents1), size(tangents2)


        kernel! = pulledback_integrand_gpu!(backend)
        KernelAbstractions.synchronize(backend)
        kernel!(tangents1, tangents2, vertices1, vertices2, I, J, ndrange = 1)
        KernelAbstractions.synchronize(backend)

        
        
        @show test_vert
        @show trail_vert
        @show test_tan
        @show trail_tan
        @show test_vol
        @show trail_vol
        @show vertices1
        @show vertices2
        @show tangents1
        @show tangents2

        t_4 = time()

        r = KernelAbstractions.zeros(backend, ComplexF64, (3,3))
        tss_real = KernelAbstractions.zeros(backend, Float64, (4,4,4,4,9))
        tss_imag = KernelAbstractions.zeros(backend, Float64, (4,4,4,4,9))
        tss = KernelAbstractions.zeros(backend, ComplexF64, 9)
        
        
        if isa(rule, BEAST.SauterSchwabQuadrature.CommonFace)
            kernel! = sauterschwab_parameterized_CommonFace_gpu!(backend)
            kernel!(r, tss_real, tss_imag, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, vertices1, vertices2, tangents1, tangents2, γ, α,  ndrange = (4,4,4,4,1))
            # println("CommonFace_")
        elseif isa(rule, BEAST.SauterSchwabQuadrature.CommonEdge)
            kernel! = sauterschwab_parameterized_CommonEdge_gpu!(backend)
            kernel!(r, tss, tss_real, tss_imag, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, vertices1, vertices2, tangents1, tangents2, γ, α,  ndrange = (4,4,4,4,1))
            # println("CommonEdge_")
        elseif isa(rule, BEAST.SauterSchwabQuadrature.CommonVertex)
            kernel! = sauterschwab_parameterized_CommonVertex_gpu!(backend)
            kernel!(r, tss_real, tss_imag, qps, test_vert, trail_vert, test_tan, trail_tan, test_vol, trail_vol, vertices1, vertices2, tangents1, tangents2, γ, α,  ndrange = (4,4,4,4,1))
            # println("CommonVertex_")
        else
            println("other") #mss PositiveDistance
        end
        KernelAbstractions.synchronize(backend)
        t_5 = time()
        # @show r
        # @show tss
        B = CUDA.@sync sum(tss_real, dims=(1,2,3,4))
        C = CUDA.@sync sum(tss_imag, dims=(1,2,3,4))
        KernelAbstractions.synchronize(backend)
        t_6 = time()
        # @show B + C * im
        # if isa(rule, BEAST.SauterSchwabQuadrature.CommonEdge)
        # throw(" ")
        # end
        # throw("")

        cpu_array = Array(B + C * im)
        cpu_array = reshape(cpu_array, 3, 3)

        num_tshapes = numfunctions(test_local_space, domain(test_chart))
        num_bshapes = numfunctions(trial_local_space, domain(trial_chart))
        t_7 = time()
        # @show cpu_array
        # @show t_2 - t_1, t_3 - t_2, t_4 - t_3, t_5 - t_4, t_6 - t_5, t_7 - t_6
        # KernelAbstractions.synchronize(backend)
        # @show num_tshapes, num_bshapes
        out[1:num_tshapes, 1:num_bshapes] .+= cpu_array
        
        KernelAbstractions.synchronize(backend)
        # cpu_array = Array(r)
        # return cpu_array
    end
end

# cpu_array = ComplexF64[
#     0.0005597241044088058 - 4.354747797567774e-5im 0.00040537499033673293 - 4.343142741227216e-5im 0.00043134657399576237 - 4.349819643624242e-5im; 
#     0.00040537499033673293 - 4.343142741227216e-5im 0.0005548879654749457 - 4.3541812881171e-5im 0.00042404721649410534 - 4.3481226121797975e-5im; 
#     0.00043134657399576237 - 4.349819643624242e-5im 0.00042404721649410534 - 4.3481226121797975e-5im 0.000572088926626421 - 4.356410789971563e-5im]
