struct ElementAssemblyData
    i_start::Int
    i_end::Int
    elements_length_tuple::Tuple{Int, Int}
    elements_data::Vector
    assembly_data::Vector
    offset::Int
    pref_offset::Int
    length_return_matrix::Int
end

function create_element_assembly_data!(i::Int, indexes, offset, pref_offset, config, 
                                test_elements_data_original, test_assembly_data, trial_assembly_data,
                                trail_elements_data_gpu, trial_assembly_data_gpu, 
                                trial_elements_length, length_return_matrix)

    i_start, i_end = indexes[i] + 1, indexes[i+1]
    test_elements_length = indexes[i+1] - indexes[i]
    offset += test_elements_length

    elements_length_tuple = (test_elements_length, trial_elements_length)

    test_elements_data_gpu, test_assembly_data_gpu = load_parameters(config, 
        test_elements_data_original, test_assembly_data, (i_start:i_end, :, :))

    elements_data = [test_elements_data_gpu... , trail_elements_data_gpu...]
    assembly_gpu_data = [
        test_assembly_data_gpu[2], 
        trial_assembly_data_gpu[2], 
        test_assembly_data_gpu[1], 
        trial_assembly_data_gpu[1], 
        test_assembly_data[2][:, i_start:i_end], 
        trial_assembly_data[2]
    ]

    return ElementAssemblyData(i_start, i_end, elements_length_tuple, elements_data, assembly_gpu_data, offset, pref_offset, length_return_matrix)
end