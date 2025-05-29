function load_parameters(backend, t_elements_data_original, t_assembly_data, writeBackStrategy, range=(:,:,:))
    t_elements_data = (
        move(backend, t_elements_data_original[1][range...]),
        move(backend, t_elements_data_original[2][range...]),
        move(backend, t_elements_data_original[3][range[1]])
    )

    t_assembly_gpu_values = move(backend, t_assembly_data[1][:,range[1]])
    trial_assembly_gpu_indexes = helper(backend, t_assembly_data, writeBackStrategy, range[1])

    return [t_elements_data, [t_assembly_gpu_values, trial_assembly_gpu_indexes]]
end

function helper(backend, trial_assembly_data, ::GpuWriteBackTrueInstance, range)
    return move(backend, trial_assembly_data[2][:, range])
end

function helper(backend, trial_assembly_data, ::GpuWriteBackFalseInstance, range=(:,:,:))
    return 0
end


function load_data(backend, type, elements_length, test_elements)
    elements_vertices_matrix = Array{type}(undef, elements_length, 3, 3)
    elements_tangents_matrix = Array{type}(undef, elements_length, 3, 3)
    elements_volume_matrix = Array{type}(undef, elements_length)
    for p in 1:elements_length
        tcell = test_elements[p]
        for i in 1:3
            elements_vertices_matrix[p,:,i] = tcell.vertices[i][:]
        end
        for i in 1:2
            elements_tangents_matrix[p,:,i] = tcell.tangents[i][:]
        end
        elements_volume_matrix[p] = tcell.volume
    end

    return elements_vertices_matrix, elements_tangents_matrix, elements_volume_matrix
end



function validate_and_extract(data, elements_length)
    size_ = (1,3,elements_length)
    data_reshaped_indexes = reshape(map(x -> x[1], data.data), (size_[2], size_[3]))
    data_reshaped_values = reshape(map(x -> x[2], data.data), (size_[2], size_[3]))
    return data_reshaped_values, data_reshaped_indexes
end