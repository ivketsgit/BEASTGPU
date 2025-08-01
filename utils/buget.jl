function budget(test_elements_length, trial_elements_length, type,
    config)
    GPU_budget = config["total_GPU_budget"]

            
    calc_budget =   3 * test_elements_length * sizeof(type)                                                    # sizeof(test_assembly_gpu_values) 
                    + 3 * trial_elements_length * sizeof(type)                                                 # sizeof(trial_assembly_gpu_values) 
                    + 3 * test_elements_length * sizeof(type)                                                  # sizeof(test_assembly_gpu_indexes) 
                    + 3 * trial_elements_length * sizeof(type)                                                 # sizeof(trial_assembly_gpu_indexes)

                    + test_elements_length *  trial_elements_length * sizeof(Int8)                            # sizeof(quadrule_types_gpu)
                    + test_elements_length *  trial_elements_length * sizeof(Int8)                            # sizeof(should_calc)
                    + (3 * test_elements_length + 2 * 3 * 3 * test_elements_length                          # sizeof(womps_weights, womps_values, womps_cart)
                    + 4 * trial_elements_length + 2 * 3 * 4 * trial_elements_length) * sizeof(type)         # sizeof(wimps_weights, wimps_values, wimps_cart)
                    + 6 * 10^30                                                                                     # sizeof of return matrix on gpu for dubble Int

                    
                    +   ((3 * 2) * sizeof(type)                                                                    # sizeof(ichart1_vert) 
                    +    (3 * 2) * sizeof(type)                                                                    # sizeof(ichart2_vert) 
                    +    (2 * 2) * sizeof(type)                                                                    # sizeof(ichart1_tan) 
                    +    (2 * 2) * sizeof(type)                                                                    # sizeof(ichart2_tan) 
                    +    (4 * 2) * sizeof(type)                                                                    #sizeof(qps)
                    +    (2) * sizeof(type)                                                                          #sizeof(store_index)
                    ) * (test_elements_length * trial_elements_length)                                      # for Vertex,z Edge and Face
                    +    1 * 10^30 * 3                                                                              # sizeof of return matrix on gpu for dubble Int
end

function choose_square_submatrix_shape(H::Int, W::Int, target_size::Int)
    target_size = ceil(sqrt(target_size))
    n_rows = cld(H, target_size)  # ceil(H / target_size)
    n_cols = cld(W, target_size)  # ceil(W / target_size)

    submatrix_height = cld(H, n_rows)
    submatrix_width  = cld(W, n_cols)

    submatrix_size = max(submatrix_height, submatrix_width)

    return Int(submatrix_size), Int(n_rows), Int(n_cols)
end

function choose_square_submatrix_shape(A::Int, target_size::Int)
    H = ceil(sqrt(A))
    W = H
    target_size = ceil(sqrt(target_size))
    n_rows = cld(H, target_size)  # ceil(H / target_size)
    n_cols = cld(W, target_size)  # ceil(W / target_size)

    submatrix_height = cld(H, n_rows)
    submatrix_width  = cld(W, n_cols)

    submatrix_size = max(submatrix_height, submatrix_width)

    return Int(submatrix_size), Int(n_rows), Int(n_cols)
end