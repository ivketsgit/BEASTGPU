

    # @show Array(should_calc_gpu_) == Array(should_calc)
    # index_Dict =  Dict(SauterSchwabQuadrature.CommonVertex{Vector{Tuple{Float64, Float64}}} => 3, SauterSchwabQuadrature.CommonEdge{Vector{Tuple{Float64, Float64}}} => 4, SauterSchwabQuadrature.CommonFace{Vector{Tuple{Float64, Float64}}} => 5, BEAST.DoubleQuadRule{Vector{@NamedTuple{weight::Float64, point::CompScienceMeshes.MeshPointNM{Float64, CompScienceMeshes.Simplex{3, 2, 1, 3, Float64}, 2, 3}, value::SVector{3, @NamedTuple{value::Float64, curl::SVector{3, Float64}}}}}, Vector{@NamedTuple{weight::Float64, point::CompScienceMeshes.MeshPointNM{Float64, CompScienceMeshes.Simplex{3, 2, 1, 3, Float64}, 2, 3}, value::SVector{3, @NamedTuple{value::Float64, curl::SVector{3, Float64}}}}}} => 1)
    # index_Dict_2 =  Dict(SauterSchwabQuadrature.CommonVertex{Vector{Tuple{Float64, Float64}}} => 1, SauterSchwabQuadrature.CommonEdge{Vector{Tuple{Float64, Float64}}} => 2, SauterSchwabQuadrature.CommonFace{Vector{Tuple{Float64, Float64}}} => 3, BEAST.DoubleQuadRule{Vector{@NamedTuple{weight::Float64, point::CompScienceMeshes.MeshPointNM{Float64, CompScienceMeshes.Simplex{3, 2, 1, 3, Float64}, 2, 3}, value::SVector{3, @NamedTuple{value::Float64, curl::SVector{3, Float64}}}}}, Vector{@NamedTuple{weight::Float64, point::CompScienceMeshes.MeshPointNM{Float64, CompScienceMeshes.Simplex{3, 2, 1, 3, Float64}, 2, 3}, value::SVector{3, @NamedTuple{value::Float64, curl::SVector{3, Float64}}}}}} => 0)
    # time_dubble_forloop = @elapsed begin 
    # for (p,(tcell,tptr)) in enumerate(zip(test_elements, test_cell_ptrs))
    #     for (q,(bcell,bptr)) in enumerate(zip(trial_elements, trial_cell_ptrs))
    #         # time = @elapsed begin
    #             qrule = quadrule(biop, test_shapes, trial_shapes, p, tcell, q, bcell, qd, quadstrat)
    #             @assert(index_Dict_2[typeof(qrule)] == quadrule_types[p, q])

    #             if !(qrule isa BEAST.DoubleQuadRule)
    #                 load_data_to_custum_data!(tcell, bcell, qrule, p, q, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace, should_calc)
    #             end
    #             # end
    #         # index = index_Dict[typeof(qrule)]
    #         # counts[index] += 1

    #         # time_table[1, index] += time

    #     end
    #     done += 1
    #     new_pctg = round(Int, done / todo * 100)
    #     if new_pctg > pctg + 9
    #         myid == 1 && print(".")
    #         pctg = new_pctg
    #     end
    # end
    # end
    # @show time_dubble_forloop




    # strategy = "repair"
    # if strategy == "repair"
    #     time_1 = doubleQuadRule_3d_gpu_outside_loops_and_store!(result_1,
    #                                                             test_assembly_gpu, trial_assembly_gpu,
    #                                                             size_qrule,
    #                                                             biop,
    #                                                             womps_weights, wimps_weights, 
    #                                                             womps_values, wimps_values, 
    #                                                             womps_cart, wimps_cart)
    #     time_table[2,1] += time_1
        
    #     time_1, time_2 = doubleQuadRule_3d_gpu_repaire!(result_1,
    #                                                     SauterSchwabQuadratureCommonVertex,
    #                                                     SauterSchwabQuadratureCommonEdge,
    #                                                     SauterSchwabQuadratureCommonFace,
    #                                                     test_assembly_gpu, trial_assembly_gpu,
    #                                                     biop,
    #                                                     womps_weights, wimps_weights, 
    #                                                     womps_values, wimps_values, 
    #                                                     womps_cart, wimps_cart)
    #     time_table[1,2] += time_1
    #     time_table[2,2] += time_2
    # elseif strategy == "should calc"
    #     time_1 = doubleQuadRule_3d_gpu_outside_loops_and_store_should_calc!(result_1,
    #                                                             test_assembly_gpu, trial_assembly_gpu,
    #                                                             size_qrule,
    #                                                             biop,
    #                                                             womps_weights, wimps_weights, 
    #                                                             womps_values, wimps_values, 
    #                                                             womps_cart, wimps_cart, should_calc)
    #     time_table[2,1] += time_1
    # else
    #     throw("strat gpu doubleQuadRule not implemented")
    # end



    



    # doubleQuadRule_generic_3d_gpu_outside_loop_square_base_algorithm!(result_1,
    # test_assembly_gpu, trial_assembly_gpu,
    # size_qrule,
    # biop,
    # womps_weights, wimps_weights, 
    # womps_values, wimps_values, 
    # womps_cart, wimps_cart, InstancedoubleQuadRuleGpuStrategyShouldCalculate(), time_table, 1, (size_qrule, size_qrule),
    # should_calc)


    

    # doubleQuadRule_generic_3d_gpu_outside_loop!(result_1,
    #     test_assembly_gpu, trial_assembly_gpu,
    #     size_qrule,
    #     biop,
    #     womps_weights, wimps_weights, 
    #     womps_values, wimps_values, 
    #     womps_cart, wimps_cart, InstancedoubleQuadRuleGpuStrategyShouldCalculate(), time_table, 1, (size_qrule, size_qrule), should_calc
    # )
    # doubleQuadRule_generic_3d_gpu_outside_loop!(result_1,
    #     test_assembly_gpu, trial_assembly_gpu,
    #     size_qrule,
    #     biop,
    #     womps_weights, wimps_weights, 
    #     womps_values, wimps_values, 
    #     womps_cart, wimps_cart, InstancedoubleQuadRuleGpuStrategyShouldCalculate(), time_table, 1, (size_qrule, size_qrule), should_calc
    # )



    # doubleQuadRule!(result_1, backend, qd.tpoints, qd.bpoints, 
    #     test_assembly_gpu, trial_assembly_gpu, 
    #     size_qrule, biop,
    #     SauterSchwabQuadratureCommonVertex,
    #     SauterSchwabQuadratureCommonEdge,
    #     SauterSchwabQuadratureCommonFace,
    #     time_table)





    
    

    # @show test_vert
    # throw("qskdmlf")

    # z = KernelAbstractions.zeros(backend, ComplexF64, (size_qrule, size_qrule, 3*3))
    # time_4 = @elapsed begin
    #     z = DoubleQuadRule_3d_gpu_outside_loops.doubleQuadRule_3d_gpu_outside_loops!(z, biop, size_qrule,
    #             womps_weights, wimps_weights, 
    #             womps_values, wimps_values, 
    #             womps_cart, wimps_cart)
    # end
    # time_5 = @elapsed begin    
    #     kernel! = store_gpu(backend)
    #     kernel!(result_real, result_imag, z, test_assembly_gpu, trial_assembly_gpu, ndrange = (size_qrule, size_qrule, 3 * 3))
    #     KernelAbstractions.synchronize(backend)
    # end
    # time_table[5] += time_4
    # time_table[6] += time_5

    # @show time_1
    # @show time_table
    # @show result_1







    
# sum([0.0069613, 0.14664689999997313, 0.04857569999999901, 0.04464260000000134, 0.0020985]) + sum([22.1028277, 0.0, 6.6328849, 3.5281623, 1.7825164]) + 8.906682100000001

# z = CuArray(Array{ComplexF64}(undef, size, size, 3, 3))
# test_assembly_gpu = CuArray(validate_and_extract(test_assembly_data))    (3, 84)
# trial_assembly_gpu = CuArray(validate_and_extract(trial_assembly_data))  (3, 84)
# result_real = KernelAbstractions.zeros(backend, Float64, 44, 44)
# result_imag = KernelAbstractions.zeros(backend, Float64, 44, 44)

# kernel!(result_real, result_imag, z, test_assembly_gpu, trial_assembly_gpu, ndrange = (size,size, 3,3))
@kernel function  store_gpu(result_real, result_imag, @Const(z), @Const(m), @Const(n))
    i, j, k =  @index(Global, NTuple)
    zt = z[i, j, k]
    quotient, remainder = divrem(k-1, 3)
    @inbounds @atomic result_real[m[quotient + 1, i], n[remainder + 1, j]] += real(zt)
    @inbounds @atomic result_imag[m[quotient + 1, i], n[remainder + 1, j]] += imag(zt)
end

# kernel!(result_real, result_imag, z, test_assembly_gpu, trial_assembly_gpu, q, p, ndrange = (I, J)) (3, 3)
@kernel function  store_gpu_inloops(result_real, result_imag, @Const(z), @Const(m), @Const(n), @Const(q), @Const(p))
    k, l =  @index(Global, NTuple)
    @atomic result_real[m[k,p], n[l,q]] += real(z[k, l])
    @atomic result_imag[m[k,p], n[l,q]] += imag(z[k, l])

end



function momintegrals_gpu!(out, op,
    test_functions::BEAST.Space, test_cellptr, test_chart,
    trial_functions::BEAST.Space, trial_cellptr, trial_chart,
    quadrule)

    local_test_space = refspace(test_functions)
    local_trial_space = refspace(trial_functions)
    
    if quadrule isa SauterSchwabQuadrature.CommonVertex || quadrule isa SauterSchwabQuadrature.CommonEdge || quadrule isa SauterSchwabQuadrature.CommonFace
        Zzz.momintegrals_gpu_this!(op,
                    local_test_space, local_trial_space,
                    test_chart, trial_chart,
                    out, quadrule)
    elseif quadrule isa BEAST.DoubleQuadRule
        DoubleQuadRule_3d_gpu.doubleQuadRule_3d_gpu!(op,
            local_test_space, local_trial_space,
            test_chart, trial_chart,
            out, quadrule)
    else 
        println("false")
        throw("")
    end
end