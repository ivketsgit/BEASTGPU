include("assamble_chunk_gpu.jl") 
using Base.Threads
function assemble_gpu(operator::BEAST.AbstractOperator, test_functions, trial_functions, writeBackStrategy::GpuWriteBack;
    storage_policy = Val{:bandedstorage},
    threading = BEAST.Threading{:single},
    # long_delays_policy = LongDelays{:compress},
        quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions))

    Z_real, Z_imag, store = allocatestorage(operator, test_functions, trial_functions, storage_policy)
    
    split = false
    Z = assemble_gpu!(operator, test_functions, trial_functions, writeBackStrategy, store, threading; quadstrat, split)
    if typeof(writeBackStrategy) == GpuWriteBackTrueInstance
        return Z
    end
    real_part = Z_real()
    imag_part = Z_imag
    return real_part + imag_part *im
end


function assemble_gpu!(operator::BEAST.Operator, test_functions::BEAST.Space, trial_functions::BEAST.Space, writeBackStrategy::GpuWriteBack,
    store, threading::Type{BEAST.Threading{:single}};
    quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    split = false)
    
    @show numfunctions(test_functions)
    assemblechunk_gpu!(operator, test_functions, trial_functions, writeBackStrategy, store; quadstrat)
end

# function assemble_gpu!(operator::BEAST.Operator, test_functions::BEAST.Space, trial_functions::BEAST.Space,
#     store, threading::Type{BEAST.Threading{:single}};
#     quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
#     split = true)

#     GPU_budget = 10^30

#     test_elements_length = x 
#     trial_elements_length = y

#     @show test_functions
#     @show numfunctions(test_functions)



#     # (length_1 * elements_length_tuple[1] + 2 * 3 * length_1 * elements_length_tuple[1]
#     #  + length_2 * elements_length_tuple[2] + 2 * 3 * length_2 * elements_length_tuple[2]) * sizeof(Float64)

#     assemblechunk_gpu!(operator, test_functions, trial_functions, store; quadstrat)
# end

function assemble_gpu!(operator::BEAST.Operator, test_functions::BEAST.Space, trial_functions::BEAST.Space,
    store, threading::Type{BEAST.Threading{:multi}};
    quadstrat=defaultquadstrat(operator, test_functions, trial_functions))

    P = 8#Threads.nthreads()
    numchunks = P
    @assert numchunks >= 1
    splits = [round(Int,s) for s in range(0, stop=numfunctions(test_functions), length=numchunks+1)]

    # @show numfunctions(test_functions)
    # @show numfunctions(trial_functions)

    # @show test_functions
    Threads.@threads for i in 1:P
        lo, hi = splits[i]+1, splits[i+1]
        lo <= hi || continue
        test_functions_p = BEAST.subset(test_functions, lo:hi)

        store1 = BEAST._OffsetStore(store, lo-1, 0)
        assemblechunk_gpu!(operator, test_functions_p, trial_functions, store1, quadstrat=quadstrat)
    end 
end


function allocatestorage(operator::BEAST.AbstractOperator, test_functions, trial_functions,
    storage_trait=nothing, longdelays_trait=nothing)

    T = promote_type(
        scalartype(operator)       ,
        scalartype(test_functions) ,
        scalartype(trial_functions),
    )
    Z_real = Matrix{Float64}(undef,
        numfunctions(test_functions),
        numfunctions(trial_functions),
    )
    Z_imag = Matrix{Float64}(undef,
        numfunctions(test_functions),
        numfunctions(trial_functions),
    )
    fill!(Z_real, 0)
    fill!(Z_imag, 0)
    # store(v,m,n) = (Z[m,n] += v)
    store(v, m, n) = begin
        @atomic Z_real[m, n] += real(v)
        @atomic Z_imag[m, n] += imag(v)
    end
    return ()->Z_real, Z_imag, store
end



# defaultquadstrat(op::HelmholtzOperator2D, tfs, bfs) = DoubleNumQStrat(4,3)
# struct DoubleNumQStrat{R}
#     outer_rule::R
#     inner_rule::R
# end
# function quadrule(operator::IntegralOperator,
#     local_test_basis, local_trial_basis,
#     test_id, test_element, trial_id, trial_element,
#     quad_data, qs::DoubleNumQStrat)

#     test_quad_rules  = quad_data[1]
#     trial_quad_rules = quad_data[2]

#     DoubleQuadRule(
#         test_quad_rules[1,test_id],
#         trial_quad_rules[1,trial_id]
#     )
# end
# struct DoubleQuadRule{P,Q}
#     outer_quad_points::P
#     inner_quad_points::Q
# end   