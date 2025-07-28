include("BEAST_integralop_copy.jl")

function allocatestorage(operator::BEAST.AbstractOperator, test_functions, trial_functions,
    storage_trait=nothing, longdelays_trait=nothing)

    T = promote_type(
        scalartype(operator)       ,
        scalartype(test_functions) ,
        scalartype(trial_functions),
    )
    Z = Matrix{T}(undef,
        numfunctions(test_functions),
        numfunctions(trial_functions),
    )
    fill!(Z, 0)
    store(v,m,n) = (Z[m,n] += v)
    return ()->Z, store
end


function assemble_multi_thread(operator, test_functions, trial_functions,print_file,appendOrWrite;
    storage_policy = Val{:bandedstorage},
    quadstrat=BEAST.defaultquadstrat)

    Z, store = allocatestorage(operator, test_functions, trial_functions,
        storage_policy)
    # qs = quadstrat(operator, test_functions, trial_functions)
    assemble_multi_thread!(operator, test_functions, trial_functions,
        store,print_file,appendOrWrite; quadstrat)
    return Z()
end

struct _OffsetStore{F}
    store::F
    row_offset::Int
    col_offset::Int
end

(f::_OffsetStore)(v,m,n) = f.store(v,m + f.row_offset, n + f.col_offset)

function assemble_multi_thread!(operator::BEAST.Operator, test_functions::BEAST.Space, trial_functions::BEAST.Space,
    store,print_file,appendOrWrite;
    quadstrat=defaultquadstrat(operator, test_functions, trial_functions))

    quadstrat = quadstrat(operator, test_functions, trial_functions)

    P = Threads.nthreads()
    numchunks = P
    @assert numchunks >= 1
    splits = [round(Int,s) for s in range(0, stop=numfunctions(test_functions), length=numchunks+1)]

    Threads.@threads for i in 1:P
        lo, hi = splits[i]+1, splits[i+1]
        lo <= hi || continue
        test_functions_p = BEAST.subset(test_functions, lo:hi)

        store1 = BEAST._OffsetStore(store, lo-1, 0)
        assemblechunk_gpu!(operator, test_functions_p, trial_functions, store1,print_file,appendOrWrite, quadstrat=quadstrat)
    end 
end







function assemble_sigle_thread(operator, test_functions, trial_functions,print_file,appendOrWrite;
    storage_policy = Val{:bandedstorage},
    quadstrat=BEAST.defaultquadstrat)

    Z, store = allocatestorage(operator, test_functions, trial_functions,
        storage_policy)
    # qs = quadstrat(operator, test_functions, trial_functions)
    assemble_sigle_thread!(operator, test_functions, trial_functions,
        store,print_file,appendOrWrite; quadstrat)
    return Z()
end



function assemble_sigle_thread!(operator::BEAST.Operator, test_functions, trial_functions,
    store,print_file,appendOrWrite;
    quadstrat=BEAST.defaultquadstrat)

    # quadstrat=defaultquadstrat(operator, test_functions, trial_functions))

    quadstrat = quadstrat(operator, test_functions, trial_functions)
    assemblechunk_gpu!(operator, test_functions, trial_functions, store,print_file,appendOrWrite; quadstrat)
end


# function assemble_sigle_thread_!(operator, test_functions, trial_functions, config,
#     store;
#     quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
#     split = false)
    
#     assemblechunk_gpu!(operator, test_functions, trial_functions, config, store; quadstrat)
# end

