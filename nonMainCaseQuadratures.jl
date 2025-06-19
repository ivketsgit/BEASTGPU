include("SauterSchwab.jl")
include("load_data_into_custom_datastructs.jl")

#CustomDataStructs
include("CustomDataStructs/SauterSchwabQuadratureDataStruct.jl")

#utils
include("utils/copy_to_CPU.jl")

function nonMainCaseQuadratures!(qd, elementAssemblyData, quadrule_types_gpu, config, 
        test_elements, trial_elements, counts, biop, store,
        timingInfo)

    elements_length_tuple = 
    
    # task = Threads.@spawn begin
    backend = config.backend
    timingInfo.time_transfer_to_CPU += @elapsed begin
        quadrule_types = Array{Int8}(undef, elementAssemblyData.elements_length_tuple)
        
        quadrule_types = copy_to_CPU(quadrule_types, quadrule_types_gpu, backend, Int8, Int(round(1024 * 1024 * 100 * 1.5)), config)
        # quadrule_types = Array(quadrule_types_gpu)
        # quadrule_types = pinned_arr(quadrule_types, backend)
        # copyto!(quadrule_types, quadrule_types_gpu)
    end


    timingInfo.time_double_forloop += @elapsed begin  
        CommonVertex_data, CommonEdge_data, CommonFace_data = load_data_into_custom_datastructs(elementAssemblyData, test_elements, trial_elements, quadrule_types, counts)
    end
    
    timingInfo.time_sauter_schwab += @elapsed begin
        @sync begin
            @async SauterSchwab!(CommonVertex_data, SauterSchwabQuadrature.CommonVertex(qd.gausslegendre[1]), 
                elementAssemblyData, biop, store, config, timingInfo)

            @async SauterSchwab!(CommonEdge_data, SauterSchwabQuadrature.CommonEdge(qd.gausslegendre[2]),
                elementAssemblyData, biop, store, config, timingInfo)

            @async SauterSchwab!(CommonFace_data, SauterSchwabQuadrature.CommonFace(qd.gausslegendre[3]),
                elementAssemblyData, biop, store, config, timingInfo)
        end
    end
    # end
    # wait(task)
end