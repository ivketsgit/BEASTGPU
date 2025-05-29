include("doubleQuadRule_3d_gpu_outside_loops_and_store.jl")
include("GPU_scheduler.jl")
include("load_parameters.jl")
include("quadrule_determine.jl")
include("load_data_into_custom_datastructs.jl")
include("SauterSchwab.jl")

#CustomDataStructs
include("CustomDataStructs/SauterSchwabQuadratureDataStruct.jl")
include("CustomDataStructs/doubleQuadRuleGpuStrategy.jl")
include("CustomDataStructs/should_calc.jl")


#utils
include("utils/noSwitch.jl")
include("utils/copy_to_CPU.jl")
include("utils/log.jl")
include("utils/move.jl")

using StaticArrays
using CompScienceMeshes   
using SauterSchwabQuadrature
using BEAST: DoubleQuadRule
using CUDA

using KernelAbstractions, Atomix
using KernelAbstractions: @atomic

# using BEAST.space
# include("../src/quadrature/sauterschwabints.jl") 
# struct DoubleQuadRule{P,Q}
#     outer_quad_points::P
#     inner_quad_points::Q
#   end
abstract type AbstractOperator end
abstract type Operator <: AbstractOperator end
abstract type IntegralOperator <: Operator end
abstract type MaxwellOperator3D{T,K} <: IntegralOperator end
abstract type Helmholtz3DOp{T,K} <: MaxwellOperator3D{T,K} end
# abstract type Helmholtz3DOp{T,K} <: BEAST.MaxwellOperator3D{T,K} end


struct HH3DSingleLayerFDBIO{T,K} <: Helmholtz3DOp{T,K}
    alpha::T
    gamma::K
end

function assemblechunk_gpu!(biop::BEAST.IntegralOperator, tfs::BEAST.Space, bfs::BEAST.Space, configuration, store;
    quadstrat=BEAST.defaultquadstrat(biop, tfs, bfs))

    tr = BEAST.assemblydata(tfs); tr == nothing && return
    br = BEAST.assemblydata(bfs); br == nothing && return

    test_elements, tad, tcells = tr
    bsis_elements, bad, bcells = br

    tgeo = geometry(tfs)
    bgeo = geometry(bfs)

    tdom = domain(chart(tgeo, first(tgeo)))
    bdom = domain(chart(bgeo, first(bgeo)))

    tshapes = refspace(tfs); num_tshapes = numfunctions(tshapes, tdom)
    bshapes = refspace(bfs); num_bshapes = numfunctions(bshapes, bdom)

    qs = if CompScienceMeshes.refines(tgeo, bgeo)
        TestRefinesTrialQStrat(quadstrat)
    elseif CompScienceMeshes.refines(bgeo, tgeo)
        TrialRefinesTestQStrat(quadstrat)
    else
        quadstrat
    end

    qd = quaddata(biop, tshapes, bshapes, test_elements, bsis_elements, qs)
    zlocal = zeros(BEAST.scalartype(biop, tfs, bfs), 2num_tshapes, 2num_bshapes)
    assemblechunk_body_gpu!(biop,
        tfs, test_elements, tad, tcells,
        bfs, bsis_elements, bad, bcells,
        qd, zlocal, configuration, store; quadstrat=qs)
end


function assemblechunk_body_gpu!(biop,
    test_space, test_elements, test_assembly_data, test_cell_ptrs,
    trial_space, trial_elements, trial_assembly_data, trial_cell_ptrs,
    qd, zlocal, configuration, store; quadstrat)
    type = Float64

    time_all = @elapsed begin
        time_overhead = @elapsed begin

            #BEAST.jl loads
            test_elements_length = length(test_elements)
            trial_elements_length = length(trial_elements)

            test_shapes = refspace(test_space)
            trial_shapes = refspace(trial_space)

            todo, done, pctg = length(test_elements), 0, 0
            length_return_matrix = length(test_space.geo.vertices)

            
            writeBackStrategy = configuration["writeBackStrategy"]
            backend = configuration["backend"]
            amount_of_gpus = configuration["amount_of_gpus"]
            InstancedoubleQuadRuleGpuStrategyShouldCalculate = configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"]
            ShouldCalcInstance = configuration["ShouldCalcInstance"]

            range_test = 1:amount_of_gpus
            range_trail = 1:1

            empty!(gpu_results_cache)

            indexes = [round(Int,s) for s in range(0, stop=length(test_elements), length=amount_of_gpus+1)]
            lengths = ones(Int, amount_of_gpus)
            for i in 1:amount_of_gpus
                lengths[i] = indexes[i+1] - indexes[i]
            end
            
            
            test_elements_data_original = load_data(backend, type, test_elements_length, test_elements)
            trial_elements_data_original = load_data(backend, type, trial_elements_length, trial_elements)

            test_assembly_data = validate_and_extract(test_assembly_data, test_elements_length)
            trial_assembly_data = validate_and_extract(trial_assembly_data, trial_elements_length)

            trail_elements_data_gpu, trial_assembly_data_gpu = load_parameters(backend, trial_elements_data_original, trial_assembly_data, writeBackStrategy)
            
            #timers
            counts = zeros(4)
            time_table = [Atomic{Float64}(0.0) for _ in 1:2, _ in 1:4]

            time_to_store = Threads.Atomic{Float64}(0)

            offset = 0
            pref_offset = 0
        
            time_double_int = 0
            time_quadrule_types = 0
            time_double_forloop = 0
            time_sauter_schwab = 0
            time_transfer_to_CPU = 0
        end
        

        for i in range_test
            #  for j in range_trail

            @sync begin
                i_start, i_end = indexes[i]+1, indexes[i+1]
                test_elements_length_ = lengths[i]
                offset += test_elements_length_
                
                elements_length_tuple = (test_elements_length_, trial_elements_length)

                test_elements_data_gpu, test_assembly_data_gpu = load_parameters(backend, test_elements_data_original, test_assembly_data, writeBackStrategy, (i_start:i_end,:,:))

                elements_data = [test_elements_data_gpu..., trail_elements_data_gpu...]
                assembly_gpu_data = [test_assembly_data_gpu[2], trial_assembly_data_gpu[2], test_assembly_data_gpu[1], trial_assembly_data_gpu[1], test_assembly_data[2][:,i_start:i_end], trial_assembly_data[2]]
                data = [elements_data, assembly_gpu_data]
                
                time_quadrule_types += @elapsed begin   
                    quadrule_types_gpu = KernelAbstractions.allocate(backend, Int8, elements_length_tuple)
                    
                    abs2_mul_16 = abs2(biop.gamma) * 16
                    floatmax_type = floatmax(type)
                    trial_elements_vertices_matrix =  trail_elements_data_gpu[1]
                    trial_elements_volume_matrix = trail_elements_data_gpu[3]
                    test_elements_vertices_matrix = test_elements_data_gpu[1]
                    # sigma_volume = KernelAbstractions.allocate(backend, eltype(trial_elements_volume_matrix), size(trial_elements_volume_matrix))

                    # permake_array_volume(backend)(sigma_volume, abs2_mul_16, trial_elements_volume_matrix, ndrange = trial_elements_length)
                    # quadrule_determine_type(backend)(quadrule_types_gpu, test_elements_vertices_matrix, trial_elements_vertices_matrix, sigma_volume, floatmax_type, ndrange = elements_length_tuple)
                    quadrule_determine_type(backend, 1024)(quadrule_types_gpu, abs2_mul_16, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatmax_type, ndrange = elements_length_tuple)
                    KernelAbstractions.synchronize(backend)
                end

                
                @async begin
                    time_transfer_to_CPU += @elapsed begin
                        quadrule_types = Array{Int8}(undef, elements_length_tuple)
                        
                        quadrule_types = copy_to_CPU(quadrule_types, quadrule_types_gpu, backend, Int8, 1024 * 1024)
                        # quadrule_types = pinned_arr(quadrule_types, backend)
                        # copyto!(quadrule_types, quadrule_types_gpu)
                    end


                    time_double_forloop += @elapsed begin  
                        CommonVertex_data, CommonEdge_data, CommonFace_data = load_data_into_custom_datastructs(test_elements_length_, pref_offset, trial_elements_length, test_elements, trial_elements, quadrule_types, counts)
                    end
                    
                    time_sauter_schwab += @elapsed begin
                        @sync begin
                            @async SauterSchwab!(CommonVertex_data, SauterSchwabQuadrature.CommonVertex(qd.gausslegendre[1]), 
                                data, biop, time_table,
                                store, length_return_matrix, time_to_store, elements_length_tuple, configuration)

                            @async SauterSchwab!(CommonEdge_data, SauterSchwabQuadrature.CommonEdge(qd.gausslegendre[2]),
                                data, biop, time_table,
                                store, length_return_matrix, time_to_store, elements_length_tuple, configuration)

                            @async SauterSchwab!(CommonFace_data, SauterSchwabQuadrature.CommonFace(qd.gausslegendre[3]),
                                data, biop, time_table,
                                store, length_return_matrix, time_to_store, elements_length_tuple, configuration)
                        end
                    end
                    @show time_sauter_schwab
                end
                
                # @async begin
                    time_double_int += @elapsed begin
                        schedule_kernel!(
                            backend, 
                            length_return_matrix, elements_length_tuple,
                            data, biop, quadrule_types_gpu, qd, type, store,
                            time_table, time_to_store, pref_offset,
                            configuration,
                            configuration["writeBackStrategy"]
                        )  
                    end
                    @show time_double_int
                # end
                

                
                pref_offset = offset

                # end
            end
            # end
        end 

        time_log = @elapsed begin
            if isdefined(Main, :time_logger)
                log_time(time_logger, "time overhead", time_overhead)
                log_time(time_logger, "time to determin the quadrule", time_quadrule_types)
                log_time(time_logger, "calculate the double int", time_double_int)
                log_time(time_logger, "transfer quadrule to CPU", time_transfer_to_CPU)
                log_time(time_logger, "calculate double for loop", time_double_forloop)
                log_time(time_logger, "calculate SauterSchwab", time_sauter_schwab)
                for i in 2:4
                    log_time(time_logger, "time_sauter_schwab_overhead_and_test_toll $i", time_table[1,i])
                    log_time(time_logger, "calc_sauter_schwab $i", time_table[2,i])
                end
                log_time(time_logger, "time_table[1,:]", time_table[1,:])
                log_time(time_logger, "time_table[2,:]", time_table[2,:])
                log_time(time_logger, "time_to_store", time_to_store)
            end
        end
        @show time_log
    end
    @show time_all
end
