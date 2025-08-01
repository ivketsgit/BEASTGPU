using KernelAbstractions
using KernelAbstractions: @atomic
using KernelAbstractions.Extras: @unroll

function reduce_(r, igd_Integrands, i)
    tot = 256
    # ((63 - leading_zeros(256)))
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

function reduce(igd_Integrands, i)
    tot = 256
    # ((63 - leading_zeros(256)))
    @unroll for iter in 8:-1:1
        d = 2 ^ (iter-1)
        if i <= d
            for local_iter in 1:9
                igd_Integrands[i, local_iter] += igd_Integrands[i + d, local_iter]
            end
        end
        @synchronize()
    end
end

const tot = 256
@inline function reduce_attomic(igd_Integrands, i)
    # ((63 - leading_zeros(256)))
    @synchronize()
    @unroll for iter in 8-3:-1:1
        d = 2 ^ (iter-1)
        if i + 1 <= d
            @unroll for local_iter in 1:18
                igd_Integrands[i * 9 * 2 + local_iter] += igd_Integrands[(i + d) * 9 * 2 + local_iter]
            end
        end
        @synchronize()
    end
end

const tot = 256
@inline function reduce_attomic__(igd_Integrands, i)
    # ((63 - leading_zeros(256)))
    @synchronize()
    for iter in 8:-1:1
        d = 2 ^ (iter-1)
        if i + 1 <= d
            for local_iter in 1:18
                igd_Integrands[(local_iter - 1) * 256 + i + 1] += igd_Integrands[(local_iter - 1) * 256 + (i + d) + 1]
            end
        end
        @synchronize()
    end
end

function reduce_attomic_(igd_Integrands, i)
    tot = 256
    inv_i = 256 - 1 - i
    # ((63 - leading_zeros(256)))
    @synchronize()
    # @unroll for local_iter in 0:9
        @unroll for iter in 8:-1:1
            d = 2 ^ (iter-1)
            if i + 1 <= d
                igd_Integrands[i + 1 + 0 * 256] += igd_Integrands[(i + d) + 1 + 0 * 256]
                igd_Integrands[i + 1 + 1 * 256] += igd_Integrands[(i + d) + 1 + 1 * 256]
                igd_Integrands[i + 1 + 2 * 256] += igd_Integrands[(i + d) + 1 + 2 * 256]
                igd_Integrands[i + 1 + 3 * 256] += igd_Integrands[(i + d) + 1 + 3 * 256]
                igd_Integrands[i + 1 + 4 * 256] += igd_Integrands[(i + d) + 1 + 4 * 256]
                igd_Integrands[i + 1 + 5 * 256] += igd_Integrands[(i + d) + 1 + 5 * 256]
                igd_Integrands[i + 1 + 6 * 256] += igd_Integrands[(i + d) + 1 + 6 * 256]
                igd_Integrands[i + 1 + 7 * 256] += igd_Integrands[(i + d) + 1 + 7 * 256]
                igd_Integrands[i + 1 + 8 * 256] += igd_Integrands[(i + d) + 1 + 8 * 256]
                igd_Integrands[i + 1 + 9 * 256] += igd_Integrands[(i + d) + 1 + 9 * 256]
            end
            if inv_i + 1 <= d
                igd_Integrands[i + 1 + 10 * 256] += igd_Integrands[(i + d) + 1 + 10 * 256]
                igd_Integrands[i + 1 + 11 * 256] += igd_Integrands[(i + d) + 1 + 11 * 256]
                igd_Integrands[i + 1 + 12 * 256] += igd_Integrands[(i + d) + 1 + 12 * 256]
                igd_Integrands[i + 1 + 13 * 256] += igd_Integrands[(i + d) + 1 + 13 * 256]
                igd_Integrands[i + 1 + 14 * 256] += igd_Integrands[(i + d) + 1 + 14 * 256]
                igd_Integrands[i + 1 + 15 * 256] += igd_Integrands[(i + d) + 1 + 15 * 256]
                igd_Integrands[i + 1 + 16 * 256] += igd_Integrands[(i + d) + 1 + 16 * 256]
                igd_Integrands[i + 1 + 17 * 256] += igd_Integrands[(i + d) + 1 + 17 * 256]

            end
            @synchronize()
        end
    # end
    # inv_i = 256 - 1 - i
    # @unroll for local_iter in 10:17
    #     @unroll for iter in 8:-1:1
    #         d = 2 ^ (iter-1)
    #         if inv_i + 1 <= d
    #             igd_Integrands[inv_i + 1 + local_iter * 256] += igd_Integrands[inv_i + d + 1 + local_iter * 256]

    #         end
    #         @synchronize()
    #     end
    # end
end



# @kernel function test_reduce_funtion(A, B)
#     # igd_Integrands = @localmem Float64 (4*4*4*4* 9 * 2)
#     Local_number = @index(Local, Linear)
#     Local_number_offset = Local_number - 1
#     reduce_attomic(B, Local_number_offset)
#     @unroll for local_iter in 1:18
#         @atomic A[local_iter] += B[local_iter]
#     end
# end


# include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# B = KernelAbstractions.ones(backend, Float64, 4*4*4*4* 9 * 2)
# A = KernelAbstractions.zeros(backend, Float64, 4*4*4*4* 9 * 2)
# kernel = test_reduce_funtion(backend, 256)
# kernel(A, B)
# @show sum(Array(A))
