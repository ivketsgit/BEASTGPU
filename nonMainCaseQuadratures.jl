include("SauterSchwab.jl")
include("load_data_into_custom_datastructs.jl")

#CustomDataStructs
include("CustomDataStructs/SauterSchwabQuadratureDataStruct.jl")

#utils
include("utils/copy_to_CPU.jl")

using KernelAbstractions
using Atomix

@kernel function sort_data(CommonVertex_data, CommonEdge_data, CommonFace_data, counters, pref_offset, @Const(quadrule_types_gpu))
    p, q = @index(Global, NTuple)

    if quadrule_types_gpu[p, q] == 1
        index = (@atomic counters[1] += 1) - 1
        # @print(" ", index)
        CommonVertex_data[index, 1] = p + pref_offset
        CommonVertex_data[index, 2] = q
    elseif quadrule_types_gpu[p, q] == 2
        index = (@atomic counters[2] += 1) - 1
        # @print(" ", index)
        CommonEdge_data[index, 1] = p + pref_offset
        CommonEdge_data[index, 2] = q
    elseif quadrule_types_gpu[p, q] == 3   
        # @print(" ", index)
        index = (@atomic counters[3] += 1) - 1
        CommonFace_data[index, 1] = p + pref_offset
        CommonFace_data[index, 2] = q
    end
end

function nonMainCaseQuadratures!(qd, elementAssemblyData, quadrule_types_gpu, config, 
        test_elements, trial_elements, counts, biop, store, sizes,
        timingInfo)
    
    # task = Threads.@spawn begin
    backend = config.backend

    if config.sortOnCPU == true
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
            task1 = Threads.@spawn SauterSchwab!(CommonVertex_data, SauterSchwabQuadrature.CommonVertex(qd.gausslegendre[1]), 
                elementAssemblyData, biop, store, config, timingInfo)

            task2 = Threads.@spawn SauterSchwab!(CommonEdge_data, SauterSchwabQuadrature.CommonEdge(qd.gausslegendre[2]),
                elementAssemblyData, biop, store, config, timingInfo)

            task3 = Threads.@spawn SauterSchwab!(CommonFace_data, SauterSchwabQuadrature.CommonFace(qd.gausslegendre[3]),
                elementAssemblyData, biop, store, config, timingInfo)
            
            wait(task1)
            wait(task2)
            wait(task3)
        end

    else

        sizes = Array(sizes)

        CommonVertex_data_ = KernelAbstractions.allocate(backend, Int64, sizes[1], 2)
        CommonEdge_data_ = KernelAbstractions.allocate(backend, Int64, sizes[2], 2)
        CommonFace_data_ = KernelAbstractions.allocate(backend, Int64, sizes[3], 2)


        pref_offset = elementAssemblyData.pref_offset

        counters = KernelAbstractions.ones(backend, Int64, 3)
        time_sort = @elapsed begin
            sort_data(backend)(CommonVertex_data_, CommonEdge_data_, CommonFace_data_, counters, pref_offset, quadrule_types_gpu, ndrange = elementAssemblyData.elements_length_tuple)
            KernelAbstractions.synchronize(backend)
        end

        timingInfo.time_sauter_schwab += @elapsed begin
                
            CommonVertex_data = SauterSchwabQuadrature_gpu_data{CommonVertexCustomGpuData}()
            CommonEdge_data = SauterSchwabQuadrature_gpu_data{CommonEdgeCustomGpuData}()
            CommonFace_data = SauterSchwabQuadrature_gpu_data{CommonFaceCustomGpuData}()


            task1 = Threads.@spawn SauterSchwab_2!(CommonVertex_data, CommonVertex_data_, sizes[1], SauterSchwabQuadrature.CommonVertex(qd.gausslegendre[1]), elementAssemblyData, biop, store, config, timingInfo)
            task2 = Threads.@spawn SauterSchwab_2!(CommonEdge_data, CommonEdge_data_, sizes[2], SauterSchwabQuadrature.CommonEdge(qd.gausslegendre[2]), elementAssemblyData, biop, store, config, timingInfo)
            task3 = Threads.@spawn SauterSchwab_2!(CommonFace_data, CommonFace_data_, sizes[3], SauterSchwabQuadrature.CommonFace(qd.gausslegendre[3]), elementAssemblyData, biop, store, config, timingInfo)
            
            wait(task1)
            wait(task2)
            wait(task3)
        end
    end

    


    
    if config.filename_benchmark != ""
        # f = function()
        #     if config.sortOnCPU == true
        #         timingInfo.time_transfer_to_CPU += @elapsed begin
        #             quadrule_types = Array{Int8}(undef, elementAssemblyData.elements_length_tuple)
                    
        #             quadrule_types = copy_to_CPU(quadrule_types, quadrule_types_gpu, backend, Int8, Int(round(1024 * 1024 * 100 * 1.5)), config)
        #         end
        #         @show timingInfo.time_transfer_to_CPU

        #         timingInfo.time_double_forloop += @elapsed begin  
        #             CommonVertex_data, CommonEdge_data, CommonFace_data = load_data_into_custom_datastructs(elementAssemblyData, test_elements, trial_elements, quadrule_types, counts)
        #         end
        #         @show timingInfo.time_double_forloop

        #     else
        #         sizes = Array(sizes)

        #         CommonVertex_data_ = KernelAbstractions.allocate(backend, Int64, sizes[1], 2)
        #         CommonEdge_data_ = KernelAbstractions.allocate(backend, Int64, sizes[2], 2)
        #         CommonFace_data_ = KernelAbstractions.allocate(backend, Int64, sizes[3], 2)


        #         pref_offset = elementAssemblyData.pref_offset

        #         counters = KernelAbstractions.ones(backend, Int64, 3)
        #         time_sort = @elapsed begin
        #             sort_data(backend)(CommonVertex_data_, CommonEdge_data_, CommonFace_data_, counters, pref_offset, quadrule_types_gpu, ndrange = elementAssemblyData.elements_length_tuple)
        #             KernelAbstractions.synchronize(backend)
        #         end
        #     end
        # end

        # manual_benchmark(f, n=300,filename=config.filename_benchmark*".txt", appendOrWrite="a")


        f = function()
            timingInfo.time_transfer_to_CPU += @elapsed begin
                quadrule_types = Array{Int8}(undef, elementAssemblyData.elements_length_tuple)
                
                quadrule_types = copy_to_CPU(quadrule_types, quadrule_types_gpu, backend, Int8, Int(round(1024 * 1024 * 100 * 1.5)), config)
            end
        end
        manual_benchmark(f, n=100,filename=config.filename_benchmark*"SortCPU_Copy.txt", appendOrWrite="a")
        f = function()
            timingInfo.time_double_forloop += @elapsed begin  
                CommonVertex_data, CommonEdge_data, CommonFace_data = load_data_into_custom_datastructs(elementAssemblyData, test_elements, trial_elements, quadrule_types, counts)
            end
        end
        manual_benchmark(f, n=100,filename=config.filename_benchmark*"SortCPU_process.txt", appendOrWrite="a")

    end
    
    # end
    # wait(task)
end