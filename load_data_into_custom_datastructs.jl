function load_data_into_custom_datastructs(elementAssemblyData, test_elements, trial_elements, quadrule_types, counts)
    test_elements_length_ = elementAssemblyData.elements_length_tuple[1]
    trial_elements_length = elementAssemblyData.elements_length_tuple[2]
    pref_offset = elementAssemblyData.pref_offset

    array = []
    threads_array = []
    nthreads = Threads.nthreads()
    indexes_ = [round(Int,s) for s in range(0, stop=test_elements_length_, length=nthreads+1)]
    for i in 1:nthreads
        SauterSchwabQuadratureCommonVertex_ = SauterSchwabQuadrature_gpu_data{CommonVertexCustomGpuData}()
        SauterSchwabQuadratureCommonEdge_ = SauterSchwabQuadrature_gpu_data{CommonEdgeCustomGpuData}()
        SauterSchwabQuadratureCommonFace_ = SauterSchwabQuadrature_gpu_data{CommonFaceCustomGpuData}()
        counts_ = zeros(3)
        
        push!(array, [[SauterSchwabQuadratureCommonVertex_, SauterSchwabQuadratureCommonEdge_, SauterSchwabQuadratureCommonFace_], counts_, [indexes_[i]+1, indexes_[i+1]]])
    end

    for i in 1:nthreads
        entry = array[i]
        job = Threads.@spawn sort_quadrule_into_custom_datastruct(entry[1][1], entry[1][2], entry[1][3], entry[2], entry[3][1], entry[3][2], pref_offset, trial_elements_length, test_elements, trial_elements, quadrule_types)
        push!(threads_array, job)
    end

    for i in 1:nthreads
        wait(threads_array[i])
    end
    
    CommonVertex_data = merge_into_first!([i[1][1] for i in array])
    CommonEdge_data = merge_into_first!([i[1][2] for i in array])
    CommonFace_data = merge_into_first!([i[1][3] for i in array])

    for i in 1:nthreads
        counts[2] += array[i][2][1]
        counts[3] += array[i][2][2]     
        counts[4] += array[i][2][3]
    end

    counts[1] += test_elements_length_*trial_elements_length - (counts[2] + counts[3] + counts[4])
    
    return CommonVertex_data, CommonEdge_data, CommonFace_data
end

function sort_quadrule_into_custom_datastruct(SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace, 
    counts, start, stop,
    pref_offset, trial_elements_length, test_elements, trial_elements, quadrule_types)

    for q in 1:trial_elements_length
        for p in start:stop
            qt = quadrule_types[p, q]
            if qt != 0
                cpu_data = transformHitsToSauterSchwabQuadrature(qt, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
                add_element(cpu_data, SVector(p, q))
            end
        end
    end
end

@inline function transformHitsToSauterSchwabQuadrature(hits, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
    hits == 1 && return SauterSchwabQuadratureCommonVertex
    hits == 2 && return SauterSchwabQuadratureCommonEdge
    hits == 3 && return SauterSchwabQuadratureCommonFace
end



const CommonVertexNumber = 1
const CommonEdgeNumber = 2
const CommonFaceNumber = 3
function load_data_to_custum_data!(tcell, bcell, qrule_and_datastorages, p, q)
    qrule = qrule_and_datastorages[1]
    cpu_data = qrule_and_datastorages[2]

    I, J, _, _ = SauterSchwabQuadrature.reorder(
        vertices(tcell),
        vertices(bcell), qrule)

    dom1 = domain(tcell)
    dom2 = domain(bcell)

    ichart1 = CompScienceMeshes.permute_vertices(dom1, I)
    ichart2 = CompScienceMeshes.permute_vertices(dom2, J)


    add_element(cpu_data, 
    ichart1.vertices, ichart2.vertices,
    ichart1.tangents, ichart2.tangents,
    ichart1.volume, ichart2.volume,

    SVector{4}(map(t -> SVector{2}(t...), qrule.qps)), SVector{2, Int64}(p, q)
    ) 
end