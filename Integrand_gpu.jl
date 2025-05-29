using KernelAbstractions: @atomic
using KernelAbstractions

const inv_4pi = 1/(4pi)  
function Integrand__mul_gpu!(i, j, k, l, igd_Integrands, u_, v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, mul_)

    bary_x_1 = fc1v[3, 1] + fc1t[1, 1] * u_[1] + fc1t[2, 1] * u_[2]
    bary_x_2 = fc1v[3, 2] + fc1t[1, 2] * u_[1] + fc1t[2, 2] * u_[2]
    
    cart_x_1 = itecv[1, 3] + itect[1, 1] * bary_x_1 + itect[1, 2] * bary_x_2
    cart_x_2 = itecv[2, 3] + itect[2, 1] * bary_x_1 + itect[2, 2] * bary_x_2
    cart_x_3 = itecv[3, 3] + itect[3, 1] * bary_x_1 + itect[3, 2] * bary_x_2

    bary_y_1 = fc2v[3, 1] + fc2t[1, 1] * v_[1] + fc2t[2, 1] * v_[2]
    bary_y_2 = fc2v[3, 2] + fc2t[1, 2] * v_[1] + fc2t[2, 2] * v_[2]

    cart_y_1 = itrcv[1, 3] + itrct[1, 1] * bary_y_1 + itrct[1, 2] * bary_y_2
    cart_y_2 = itrcv[2, 3] + itrct[2, 1] * bary_y_1 + itrct[2, 2] * bary_y_2
    cart_y_3 = itrcv[3, 3] + itrct[3, 1] * bary_y_1 + itrct[3, 2] * bary_y_2

    R = @fastmath sqrt((cart_x_1 - cart_y_1)^2 + (cart_x_2 - cart_y_2)^2 + (cart_x_3 - cart_y_3)^2)
    green = @fastmath exp(-R*im) * inv_4pi / R

    f_1 = bary_x_1
    f_2 = bary_x_2
    f_3 = 1 - bary_x_1 - bary_x_2
    g_1 = bary_y_1
    g_2 = bary_y_2
    g_3 = 1 - bary_y_1 - bary_y_2

    jacobian_x_mul_jacobian_y_mul_green = 4 * itecvo[1] * itrcvo[1] * green * mul_
    
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


@inline function calc_bary(u_, fcv, fct)
    bary_1 = fcv[3, 1] + fct[1, 1] * u_[1] + fct[2, 1] * u_[2]
    bary_2 = fcv[3, 2] + fct[1, 2] * u_[1] + fct[2, 2] * u_[2]
    return (bary_1, bary_2, 1 - bary_1 - bary_2)
end


@inline function Integrand__mul_gpu_attomic!(Local_number, igd_Integrands, u_, v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mul_)

    bary_x = calc_bary(u_, fc1v, fc1t)
    bary_y = calc_bary(v_, fc2v, fc2t)


    # cart_x_1 = itecv[1, 3] + itect[1, 1] * bary_x[1] + itect[1, 2] * bary_x[2]
    # cart_x_2 = itecv[2, 3] + itect[2, 1] * bary_x[1] + itect[2, 2] * bary_x[2]
    # cart_x_3 = itecv[3, 3] + itect[3, 1] * bary_x[1] + itect[3, 2] * bary_x[2]


    # cart_y_1 = itrcv[1, 3] + itrct[1, 1] * bary_y[1] + itrct[1, 2] * bary_y[2]
    # cart_y_2 = itrcv[2, 3] + itrct[2, 1] * bary_y[1] + itrct[2, 2] * bary_y[2]
    # cart_y_3 = itrcv[3, 3] + itrct[3, 1] * bary_y[1] + itrct[3, 2] * bary_y[2]

    # R = sqrt((cart_x_1 - cart_y_1)^2 + (cart_x_2 - cart_y_2)^2 + (cart_x_3 - cart_y_3)^2) #

    # R = 0
    # d = itecv[1, 3] + itect[1, 1] * bary_x[1] + itect[1, 2] * bary_x[2] - (itrcv[1, 3] + itrct[1, 1] * bary_y[1] + itrct[1, 2] * bary_y[2])
    # R += d^2
    # d = itecv[2, 3] + itect[2, 1] * bary_x[1] + itect[2, 2] * bary_x[2] - (itrcv[2, 3] + itrct[2, 1] * bary_y[1] + itrct[2, 2] * bary_y[2])
    # R += d^2
    # d = itecv[3, 3] + itect[3, 1] * bary_x[1] + itect[3, 2] * bary_x[2] - (itrcv[3, 3] + itrct[3, 1] * bary_y[1] + itrct[3, 2] * bary_y[2])
    # R += d^2

    # R = sqrt(R)

    cart_x_1 = itecv[1,3] + itect[1,1]*bary_x[1] + itect[1,2]*bary_x[2]
    cart_x_2 = itecv[2,3] + itect[2,1]*bary_x[1] + itect[2,2]*bary_x[2]
    cart_x_3 = itecv[3,3] + itect[3,1]*bary_x[1] + itect[3,2]*bary_x[2]

    cart_y_1 = itrcv[1,3] + itrct[1,1]*bary_y[1] + itrct[1,2]*bary_y[2]
    cart_y_2 = itrcv[2,3] + itrct[2,1]*bary_y[1] + itrct[2,2]*bary_y[2]
    cart_y_3 = itrcv[3,3] + itrct[3,1]*bary_y[1] + itrct[3,2]*bary_y[2]

    dx1 = cart_x_1 - cart_y_1
    dx2 = cart_x_2 - cart_y_2
    dx3 = cart_x_3 - cart_y_3

    R = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3)

    green = inv_4pi / R #
    green_2 = exp(-R*γ) 

    # f_1 = bary_x[1]
    # f_2 = bary_x[2]
    # f_3 = bary_x[3]
    # f = (bary_x[1], bary_x[2], 1 - bary_x[1] - bary_x[2])
    # g = (bary_y[1], bary_y[2], 1 - bary_y[1] - bary_y[2])
    # g_1 = bary_y[1]
    # g_2 = bary_y[2]
    # g_3 = bary_y[3]

    jacobian_x_mul_jacobian_y_mul_green = α * 4 * itecvo[1] * itrcvo[1] * green * mul_

    # base_idx = Local_number * 9 * 2
    # base_idx = 0
    # for i in 0:2
    #     jacobian_x_mul_jacobian_y_mul_green_g = jacobian_x_mul_jacobian_y_mul_green * bary_y[i+1]
    #     for j in 1:3
    #         idx = base_idx + i * 6 + 2 * (j - 1) + 1
    #         @atomic igd_Integrands[idx    ] += jacobian_x_mul_jacobian_y_mul_green_g * bary_x[j] * real(green_2)
    #         @atomic igd_Integrands[idx + 1] += jacobian_x_mul_jacobian_y_mul_green_g * bary_x[j] * imag(green_2)
    #     end
    # end


    for i in 1:3
        jacobian_x_mul_jacobian_y_mul_green_g = jacobian_x_mul_jacobian_y_mul_green * bary_y[i]
        for j in 1:3
            idx = (i-1)*256*3*2 + (j-1) *256*2
            igd_Integrands[idx       + Local_number + 1] += jacobian_x_mul_jacobian_y_mul_green_g * bary_x[j] * real(green_2)
            igd_Integrands[idx + 256 + Local_number + 1] += jacobian_x_mul_jacobian_y_mul_green_g * bary_x[j] * imag(green_2)
        end
    end
    
    # jacobian_x_mul_jacobian_y_mul_green_f_1 = jacobian_x_mul_jacobian_y_mul_green * f_1
    # jacobian_x_mul_jacobian_y_mul_green_f_2 = jacobian_x_mul_jacobian_y_mul_green * f_2
    # jacobian_x_mul_jacobian_y_mul_green_f_3 = jacobian_x_mul_jacobian_y_mul_green * f_3


    # igd_Integrands[Local_number * 9 * 2 + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)
    # igd_Integrands[Local_number * 9 * 2 + 2] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)

    # igd_Integrands[Local_number * 9 * 2 + 3] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)
    # igd_Integrands[Local_number * 9 * 2 + 4] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)

    # igd_Integrands[Local_number * 9 * 2 + 5] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)
    # igd_Integrands[Local_number * 9 * 2 + 6] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)



    # igd_Integrands[Local_number * 9 * 2 + 7] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)
    # igd_Integrands[Local_number * 9 * 2 + 8] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)

    # igd_Integrands[Local_number * 9 * 2 + 9] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)
    # igd_Integrands[Local_number * 9 * 2 + 10] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)

    # igd_Integrands[Local_number * 9 * 2 + 11] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    # igd_Integrands[Local_number * 9 * 2 + 12] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    


    # igd_Integrands[Local_number * 9 * 2 + 13] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)
    # igd_Integrands[Local_number * 9 * 2 + 14] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)

    # igd_Integrands[Local_number * 9 * 2 + 15] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)
    # igd_Integrands[Local_number * 9 * 2 + 16] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)

    # igd_Integrands[Local_number * 9 * 2 + 17] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)
    # igd_Integrands[Local_number * 9 * 2 + 18] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)


    # Local_number_ = Local_number + 1
    # igd_Integrands[0 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)
    # igd_Integrands[1 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)

    # igd_Integrands[2 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)
    # igd_Integrands[3 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)

    # igd_Integrands[4 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)
    # igd_Integrands[5 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)



    # igd_Integrands[6 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)
    # igd_Integrands[7 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)

    # igd_Integrands[8 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)
    # igd_Integrands[9 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)

    # igd_Integrands[10 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    # igd_Integrands[11 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    


    # igd_Integrands[12 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)
    # igd_Integrands[13 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)

    # igd_Integrands[14 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)
    # igd_Integrands[15 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)

    # igd_Integrands[16 * 256 + Local_number_] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)
    # igd_Integrands[17 * 256 + Local_number_] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)








    # @atomic igd_Integrands[(Local_number * 9 * 2 + 1 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 2 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)

    # @atomic igd_Integrands[(Local_number * 9 * 2 + 3 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 4 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)

    # @atomic igd_Integrands[(Local_number * 9 * 2 + 5 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 6 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)



    # @atomic igd_Integrands[(Local_number * 9 * 2 + 7 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 8 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)

    # @atomic igd_Integrands[(Local_number * 9 * 2 + 9 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 10 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)

    # @atomic igd_Integrands[(Local_number * 9 * 2 + 11 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 12 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    


    # @atomic igd_Integrands[(Local_number * 9 * 2 + 13 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 14 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)

    # @atomic igd_Integrands[(Local_number * 9 * 2 + 15 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 16 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)

    # @atomic igd_Integrands[(Local_number * 9 * 2 + 17 - 1 ) % (1*18) + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)
    # @atomic igd_Integrands[(Local_number * 9 * 2 + 18 - 1 ) % (1*18) + 1] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)

    
    # igd_Integrands[Local_number * 9 * 2 + 1] = real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)
    # igd_Integrands[Local_number * 9 * 2 + 2] = imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)

    # igd_Integrands[Local_number * 9 * 2 + 3] = real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)
    # igd_Integrands[Local_number * 9 * 2 + 4] = imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)

    # igd_Integrands[Local_number * 9 * 2 + 5] = real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)
    # igd_Integrands[Local_number * 9 * 2 + 6] = imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)



    # igd_Integrands[Local_number * 9 * 2 + 7] = real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)
    # igd_Integrands[Local_number * 9 * 2 + 8] = imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)

    # igd_Integrands[Local_number * 9 * 2 + 9] = real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)
    # igd_Integrands[Local_number * 9 * 2 + 10] = imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)

    # igd_Integrands[Local_number * 9 * 2 + 11] = real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    # igd_Integrands[Local_number * 9 * 2 + 12] = imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    


    # igd_Integrands[Local_number * 9 * 2 + 13] = real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)
    # igd_Integrands[Local_number * 9 * 2 + 14] = imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)

    # igd_Integrands[Local_number * 9 * 2 + 15] = real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)
    # igd_Integrands[Local_number * 9 * 2 + 16] = imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)

    # igd_Integrands[Local_number * 9 * 2 + 17] = real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)
    # igd_Integrands[Local_number * 9 * 2 + 18] = imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)

end


function Integrand_gpu_attomic!(Local_number, igd_Integrands, u_, v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α)

    bary_x_1 = fc1v[3, 1] + fc1t[1, 1] * u_[1] + fc1t[2, 1] * u_[2]
    bary_x_2 = fc1v[3, 2] + fc1t[1, 2] * u_[1] + fc1t[2, 2] * u_[2]

    cart_x_1 = itecv[1, 3] + itect[1, 1] * bary_x_1 + itect[1, 2] * bary_x_2
    cart_x_2 = itecv[2, 3] + itect[2, 1] * bary_x_1 + itect[2, 2] * bary_x_2
    cart_x_3 = itecv[3, 3] + itect[3, 1] * bary_x_1 + itect[3, 2] * bary_x_2

    bary_y_1 = fc2v[3, 1] + fc2t[1, 1] * v_[1] + fc2t[2, 1] * v_[2]
    bary_y_2 = fc2v[3, 2] + fc2t[1, 2] * v_[1] + fc2t[2, 2] * v_[2]

    cart_y_1 = itrcv[1, 3] + itrct[1, 1] * bary_y_1 + itrct[1, 2] * bary_y_2
    cart_y_2 = itrcv[2, 3] + itrct[2, 1] * bary_y_1 + itrct[2, 2] * bary_y_2
    cart_y_3 = itrcv[3, 3] + itrct[3, 1] * bary_y_1 + itrct[3, 2] * bary_y_2

    R =  sqrt((cart_x_1 - cart_y_1)^2 + (cart_x_2 - cart_y_2)^2 + (cart_x_3 - cart_y_3)^2) #@fastmath
    green =  exp(-R*γ) * inv_4pi / R #@fastmath

    f_1 = bary_x_1
    f_2 = bary_x_2
    f_3 = 1 - bary_x_1 - bary_x_2
    g_1 = bary_y_1
    g_2 = bary_y_2
    g_3 = 1 - bary_y_1 - bary_y_2

    jacobian_x_mul_jacobian_y_mul_green = α * 4 * itecvo[1] * itrcvo[1] * green
    
    jacobian_x_mul_jacobian_y_mul_green_f_1 = jacobian_x_mul_jacobian_y_mul_green * f_1
    jacobian_x_mul_jacobian_y_mul_green_f_2 = jacobian_x_mul_jacobian_y_mul_green * f_2
    jacobian_x_mul_jacobian_y_mul_green_f_3 = jacobian_x_mul_jacobian_y_mul_green * f_3


    @atomic igd_Integrands[Local_number * 9 * 2 + 1] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)
    @atomic igd_Integrands[Local_number * 9 * 2 + 2] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)

    @atomic igd_Integrands[Local_number * 9 * 2 + 3] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)
    @atomic igd_Integrands[Local_number * 9 * 2 + 4] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)

    @atomic igd_Integrands[Local_number * 9 * 2 + 5] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)
    @atomic igd_Integrands[Local_number * 9 * 2 + 6] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)



    @atomic igd_Integrands[Local_number * 9 * 2 + 7] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)
    @atomic igd_Integrands[Local_number * 9 * 2 + 8] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)

    @atomic igd_Integrands[Local_number * 9 * 2 + 9] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)
    @atomic igd_Integrands[Local_number * 9 * 2 + 10] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)

    @atomic igd_Integrands[Local_number * 9 * 2 + 11] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    @atomic igd_Integrands[Local_number * 9 * 2 + 12] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)
    


    @atomic igd_Integrands[Local_number * 9 * 2 + 13] += real(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)
    @atomic igd_Integrands[Local_number * 9 * 2 + 14] += imag(jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)

    @atomic igd_Integrands[Local_number * 9 * 2 + 15] += real(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)
    @atomic igd_Integrands[Local_number * 9 * 2 + 16] += imag(jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)

    @atomic igd_Integrands[Local_number * 9 * 2 + 17] += real(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)
    @atomic igd_Integrands[Local_number * 9 * 2 + 18] += imag(jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)
end