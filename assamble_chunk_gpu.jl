include("GPU_scheduler.jl")
include("load_parameters.jl")
include("quadrule_determine.jl")
include("nonMainCaseQuadratures.jl")

#CustomDataStructs
include("CustomDataStructs/doubleQuadRuleGpuStrategy.jl")
include("CustomDataStructs/should_calc.jl")

#utils
include("utils/noSwitch.jl")
include("utils/log.jl")
include("utils/move.jl")
include("utils/ElementAssemblyData.jl")

using StaticArrays
using CompScienceMeshes   
using SauterSchwabQuadrature
using BEAST: DoubleQuadRule
using CUDA

###remove
using BEAST
include("utils/benchmark_alternative.jl")
#

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

function assemblechunk_gpu!(biop::BEAST.IntegralOperator, tfs::BEAST.Space, bfs::BEAST.Space, config, store;
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
        qd, zlocal, config, store; quadstrat=qs)
end

mutable struct TimingInfo
    time_overhead::Float64
    time_double_int::Float64
    time_quadrule_types::Float64
    time_double_forloop::Float64
    time_sauter_schwab::Float64
    time_transfer_to_CPU::Float64
    time_to_store::Threads.Atomic{Float64}
    time_table::Matrix{Atomic{Float64}}
end

function double_loop(test_elements, trial_elements, biop, test_space, trial_space, qd, quadstrat)  
    for (p, tcell) in enumerate(test_elements)
        for (q, bcell) in enumerate(trial_elements)
            # fill!(zlocal, 0)
            qrule = BEAST.quadrule(biop, test_space, trial_space, p, tcell, q, bcell, qd, quadstrat)
        end
    end
end


function assemblechunk_body_gpu!(biop,
    test_space, test_elements, test_assembly_data, test_cell_ptrs,
    trial_space, trial_elements, trial_assembly_data, trial_cell_ptrs,
    qd, zlocal, config, store; quadstrat)

    
    time_all = @elapsed begin
        #timers
        timingInfo = TimingInfo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Threads.Atomic{Float64}(0), [Atomic{Float64}(0.0) for _ in 1:2, _ in 1:4])

        timingInfo.time_overhead = @elapsed begin
            #BEAST.jl loads
            test_shapes = refspace(test_space)
            trial_shapes = refspace(trial_space)

            todo, done, pctg = length(test_elements), 0, 0
            length_return_matrix = length(test_space.geo.vertices)


            amount_of_gpus = config.amount_of_gpus
            range_test = 1:amount_of_gpus
            range_trail = 1:1

            empty!(gpu_results_cache)

            indexes = [round(Int,s) for s in range(0, stop=length(test_elements), length=amount_of_gpus+1)]
            trial_elements_length = length(trial_elements)
            
            test_elements_data_original = load_data_elements(config, test_elements)
            trial_elements_data_original = load_data_elements(config, trial_elements)

            test_assembly_data = extract_values_and_indexes(test_assembly_data)
            trial_assembly_data = extract_values_and_indexes(trial_assembly_data)

            trail_elements_data_gpu, trial_assembly_data_gpu = load_parameters(config, trial_elements_data_original, trial_assembly_data)

            offset = 0
            pref_offset = 0
        
            counts = zeros(4)
        end

        for i in range_test
            #  for j in range_trail
            # @sync begin
                #set data for loop
                elementAssemblyData = create_element_assembly_data!(i, indexes, offset, pref_offset, config, 
                                                                    test_elements_data_original, test_assembly_data, trial_assembly_data,
                                                                    trail_elements_data_gpu, trial_assembly_data_gpu, 
                                                                    trial_elements_length, length_return_matrix)

                #start calculations
                print("gpu")


                quadrule_types_gpu, sizes = determine_quadrule_types(config, biop, elementAssemblyData, timingInfo)
                # if config.filename_benchmark != ""
                #     manual_benchmark(determine_quadrule_types; args=(config, biop, elementAssemblyData, timingInfo), n=100,filename=config.filename_benchmark*"_determine_quadrule_types.txt", appendOrWrite="a")
                # end


                
                # @async begin
                    nonMainCaseQuadratures!(qd, elementAssemblyData, quadrule_types_gpu, config, 
                        test_elements, trial_elements, counts, biop, store, sizes,
                        timingInfo)
                # end
                
                # @async begin
                    timingInfo.time_double_int += @elapsed begin
                        schedule_kernel!(elementAssemblyData,
                            biop, quadrule_types_gpu, qd, store,
                            timingInfo, config, config.writeBackStrategy,
                        )  
                    end
                # end
                pref_offset = elementAssemblyData.offset
            # end
        end 

        log_to_config(config, timingInfo)
    end
    @show time_all
    print("\n")
end
