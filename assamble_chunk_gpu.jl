# include("../src/quadrature/doublenumints.jl")
# using .Zzz      

include("doubleQuadRule_3d_gpu_outside_loops_and_store.jl")
# using .DoubleQuadRule_3d_gpu_outside_loops_and_store
using Serialization

include("CustomDataStructs/SauterSchwabQuadratureDataStruct.jl")
include("CustomDataStructs/doubleQuadRuleGpuStrategy.jl")
include("CustomDataStructs/should_calc.jl")

include("GPU_scheduler.jl")
include("utils/noSwitch.jl")

# include("doubleQuadRule_3d_gpu_outside_loops_and_store.jl")
# using .DoubleQuadRule_3d_gpu_outside_loops_and_store


include("momintegrals_outside_loop.jl")
# include("utils/backend.jl")
# using .Momintegrals

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

function validate_and_extract(data, elements_length)
    size_ = (1,3,elements_length)
    data_reshaped_indexes = reshape(map(x -> x[1], data.data), (size_[2], size_[3]))
    data_reshaped_values = reshape(map(x -> x[2], data.data), (size_[2], size_[3]))
    return data_reshaped_values, data_reshaped_indexes
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



const dtol = 2.220446049250313e-16 * 1.0e3
const xtol2 = 0.2 * 0.2
@kernel function quadrule_determine_type(result::AbstractArray, should_calc_gpu_::AbstractArray, @Const(k2::Float64), @Const(τ::AbstractArray), @Const(σ::AbstractArray), @Const(σ_volume::AbstractArray), @Const(floatmax_type::Float64), @Const(T::ShouldCalcTrue))
    i, j = @index(Global, NTuple)

    hits = 0
    dmin2 = floatmax_type 
    @inbounds @unroll for unroll in 1:3
        @inbounds @unroll for unroll_ in 1:3
            d2 = (τ[i, 1, unroll] - σ[j, 1, unroll_])^2 + 
                 (τ[i, 2, unroll] - σ[j, 2, unroll_])^2 + 
                 (τ[i, 3, unroll] - σ[j, 3, unroll_])^2
            d = sqrt(d2)
            hits += (d < dtol)
            dmin2 = min(dmin2, d2)
        end
    end
    
    h2 = σ_volume[j]
    result[i,j] = (hits == 0 ? (max(dmin2*k2, dmin2/(16 * h2)) >= xtol2 ? 0 : 4) : hits)
    


    # index = ((i - 1) ÷ 32 ) + 1
    # bitindex = ((i - 1) % 32 ) + 1

    # mask = (hits == 0) * (1 << bitindex)
    # @print("\n UInt32(mask) = ", UInt32(mask), " ",should_calc_gpu_[index, j])
    # @print("\n UInt32(mask) = ", mask, " ",should_calc_gpu_[index, j])
    # b = should_calc_gpu_[index, j] | mask
    # @print("\n b = ", b)

    should_calc_gpu_[i, j] = hits == 0
    
    # should_calc_gpu_[index, j] |= mask
end

@kernel function quadrule_determine_type(result::AbstractArray, should_calc_gpu_::Int, @Const(k2::Float64), @Const(τ::AbstractArray), @Const(σ::AbstractArray), @Const(σ_volume::AbstractArray), @Const(floatmax_type::Float64), @Const(T::ShouldCalcFalse))
    i, j = @index(Global, NTuple)

    hits = 0
    dmin2 = floatmax_type 
    @inbounds @unroll for unroll in 1:3
        @inbounds @unroll for unroll_ in 1:3
            d2 = (τ[i, 1, unroll] - σ[j, 1, unroll_])^2 + 
                 (τ[i, 2, unroll] - σ[j, 2, unroll_])^2 + 
                 (τ[i, 3, unroll] - σ[j, 3, unroll_])^2
            d = sqrt(d2)
            hits += (d < dtol)
            dmin2 = min(dmin2, d2)
        end
    end
    
    h2 = σ_volume[j]
    result[i,j] = (hits == 0 ? (max(dmin2*k2, dmin2/(16 * h2)) >= xtol2 ? 0 : 4) : hits)
    


    # index = ((i - 1) ÷ 32 ) + 1
    # bitindex = ((i - 1) % 32 ) + 1

    # mask = (hits == 0) * (1 << bitindex)
    # @print("\n UInt32(mask) = ", UInt32(mask), " ",should_calc_gpu_[index, j])
    # @print("\n UInt32(mask) = ", mask, " ",should_calc_gpu_[index, j])
    # b = should_calc_gpu_[index, j] | mask
    # @print("\n b = ", b)
    
    # should_calc_gpu_[index, j] |= mask
end


function transformHitsToSauterSchwabQuadrature(hits, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
    hits == 1 && return SauterSchwabQuadratureCommonVertex
    hits == 2 && return SauterSchwabQuadratureCommonEdge
    hits == 3 && return SauterSchwabQuadratureCommonFace
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
    # elements_vertices_matrix = move(backend,elements_vertices_matrix)
    # elements_tangents_matrix = move(backend,elements_tangents_matrix)
    # elements_volume_matrix = move(backend,elements_volume_matrix)

    return elements_vertices_matrix, elements_tangents_matrix, elements_volume_matrix
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


function sort_quadrule_into_custom_datastruct(SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace, 
    counts, start, stop,
    pref_offet, trial_elements_length, test_elements, trial_elements, quadrule_types)

    # @show SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace
    # @show counts
    # @show start, stop

    for p in start:stop
        # tcell = test_elements[p]
        for q in 1:trial_elements_length
            qt = quadrule_types[p, q]
            if qt != 0
                # time = @elapsed begin 
                    # index = ((p - 1) ÷ 32 ) + 1
                    # bitindex = ((p - 1) % 32 ) + 1
                    # should_calc[index,q] = should_calc[index, q] & ~(1 << (bitindex - 1))

                    # should_calc[p,q] = 0

                # bcell = trial_elements[q]
                # @noswitch begin
                    
                    cpu_data = transformHitsToSauterSchwabQuadrature(qt, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
                    add_element(cpu_data, (p, q))
                end
                # end
                # @show qrule.qps
                # @show typeof(qrule.qps)
                # load_data_to_custum_data_for_gpu!(tcell, bcell, qrule_and_datastorages, p, q)
                # function load_data_to_custum_data_for_gpu!(tcell, bcell, qrule_and_datastorages, p, q)
                #     qrule = qrule_and_datastorages[1]
                #     cpu_data = qrule_and_datastorages[2]
                # end


            #     # end
            #     # time_table[1, quadrule_types[p,q] + 1] += time
            #     # if quadrule_types[p,q] != 1 && quadrule_types[p,q] != 2 && quadrule_types[p,q] != 3
            #     #     @show quadrule_types[p,q]
            #     # end
            #     # counts[quadrule_types[p,q]] += 1
            # end
            # if sum(quadrule_types[p, q:q+7]) == 0
            #     continue
            # else
            #     if quadrule_types[p, q] == 0
                    
                # elseif quadrule_types[p, q] == 1
                #     add_element(SauterSchwabQuadratureCommonVertex, [p, q])
                #     # counts[1] += 1
                # elseif quadrule_types[p, q] == 2
                #     add_element(SauterSchwabQuadratureCommonEdge, [p, q])
                #     # counts[2] += 1
                # elseif quadrule_types[p, q] == 3
                #     add_element(SauterSchwabQuadratureCommonFace, [p, q])
                #     # counts[3] += 1
                # else
                #     #call BEAST.jl momintegrals()
                #     @show quadrule_types[p,q]
                # end
            # end

        end
    end
end




function assemblechunk_body_gpu!(biop,
    test_space, test_elements, test_assembly_data, test_cell_ptrs,
    trial_space, trial_elements, trial_assembly_data, trial_cell_ptrs,
    qd, zlocal, configuration, store; quadstrat)
    type = Float64

    
    time_overhead = @elapsed begin
        test_elements_length = length(test_elements)
        trial_elements_length = length(trial_elements)

        GPU_budget = configuration["total_GPU_budget"]
        
        calc_budget =   3 * test_elements_length * sizeof(Float64)                                                    # sizeof(test_assembly_gpu_values) 
                        + 3 * trial_elements_length * sizeof(Float64)                                                 # sizeof(trial_assembly_gpu_values) 
                        + 3 * test_elements_length * sizeof(Float64)                                                  # sizeof(test_assembly_gpu_indexes) 
                        + 3 * trial_elements_length * sizeof(Float64)                                                 # sizeof(trial_assembly_gpu_indexes)

                        + test_elements_length *  trial_elements_length * sizeof(Int8)                            # sizeof(quadrule_types_gpu)
                        + test_elements_length *  trial_elements_length * sizeof(Int8)                            # sizeof(should_calc)
                        + (3 * test_elements_length + 2 * 3 * 3 * test_elements_length                          # sizeof(womps_weights, womps_values, womps_cart)
                        + 4 * trial_elements_length + 2 * 3 * 4 * trial_elements_length) * sizeof(type)         # sizeof(wimps_weights, wimps_values, wimps_cart)
                        + 6 * 10^30                                                                                     # sizeof of return matrix on gpu for dubble Int

                        
                        +   ((3 * 2) * sizeof(Float64)                                                                    # sizeof(ichart1_vert) 
                        +    (3 * 2) * sizeof(Float64)                                                                    # sizeof(ichart2_vert) 
                        +    (2 * 2) * sizeof(Float64)                                                                    # sizeof(ichart1_tan) 
                        +    (2 * 2) * sizeof(Float64)                                                                    # sizeof(ichart2_tan) 
                        +    (4 * 2) * sizeof(Float64)                                                                    #sizeof(qps)
                        +    (2) * sizeof(Int64)                                                                          #sizeof(store_index)
                        ) * (test_elements_length * trial_elements_length)                                      # for Vertex,z Edge and Face
                        +    1 * 10^30 * 3                                                                              # sizeof of return matrix on gpu for dubble Int

        
        amount_of_gpus = configuration["amount_of_gpus"]

        range_test = 1:amount_of_gpus
        range_trail = 1:1


        empty!(gpu_results_cache)


        test_shapes = refspace(test_space)
        trial_shapes = refspace(trial_space)

        todo, done, pctg = length(test_elements), 0, 0
        length_return_matrix = length(test_space.geo.vertices)

        
        #create gpu array while quadrule is running
        create_results = Threads.@spawn begin
            if typeof(configuration["writeBackStrategy"]) == GpuWriteBackTrueInstance
                result_1 = create_results_matrix_gpu(backend, length_return_matrix, (0,0), configuration["writeBackStrategy"], configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"])
            end
        end
        
        writeBackStrategy = configuration["writeBackStrategy"]

        indexes = [round(Int,s) for s in range(0, stop=length(test_elements), length=amount_of_gpus+1)]
        lengths = ones(Int, amount_of_gpus)
        for i in 1:amount_of_gpus
            lengths[i] = indexes[i+1] - indexes[i]
        end
        
        counts = zeros(4)
        time_table = [Atomic{Float64}(0.0) for _ in 1:2, _ in 1:4]
        
        InstancedoubleQuadRuleGpuStrategyShouldCalculate = configuration["InstancedoubleQuadRuleGpuStrategyShouldCalculate"]
        ShouldCalcInstance = configuration["ShouldCalcInstance"]
        
        
        test_elements_vertices_matrix_original, test_elements_tangents_matrix_original, test_elements_volume_matrix_original = load_data(backend, type, test_elements_length, test_elements)
        trial_elements_vertices_matrix_original, trial_elements_tangents_matrix_original, trial_elements_volume_matrix_original = load_data(backend, type, trial_elements_length, trial_elements)
    



        test_assembly_cpu_values, test_assembly_cpu_indexes = validate_and_extract(test_assembly_data, test_elements_length)
        trial_assembly_cpu_values, trial_assembly_cpu_indexes = validate_and_extract(trial_assembly_data, trial_elements_length)

        time_to_store = Threads.Atomic{Float64}(0)
    
        test_assembly_cpu_values_original, test_assembly_cpu_indexes_original = test_assembly_cpu_values, test_assembly_cpu_indexes
        trial_assembly_cpu_values_original, trial_assembly_cpu_indexes_original = trial_assembly_cpu_values, trial_assembly_cpu_indexes
        
        

        
        offset = 0
        pref_offet = 0
    
        time_double_int = 0
        time_quadrule_types = 0
        time_double_forloop = 0
        time_sauter_schwab = 0
        time_transfer_to_CPU = 0
    end
    

    for i in range_test; for j in range_trail

        i_start, i_end = indexes[i]+1, indexes[i+1]
        test_elements_length_ = lengths[i]
        offset += test_elements_length_

        
        
        elements_length_tuple = (test_elements_length_, trial_elements_length)

        test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix = test_elements_vertices_matrix_original[i_start:i_end,:,:], test_elements_tangents_matrix_original[i_start:i_end,:,:], test_elements_volume_matrix_original[i_start:i_end]
        test_assembly_cpu_values, test_assembly_cpu_indexes = test_assembly_cpu_values_original[:,i_start:i_end], test_assembly_cpu_indexes_original[:,i_start:i_end]

        test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix = move(backend, test_elements_vertices_matrix), move(backend, test_elements_tangents_matrix), move(backend, test_elements_volume_matrix)
        trial_elements_vertices_matrix, trial_elements_tangents_matrix, trial_elements_volume_matrix = move(backend, trial_elements_vertices_matrix_original), move(backend, trial_elements_tangents_matrix_original), move(backend, trial_elements_volume_matrix_original)


        test_assembly_gpu_values = move(backend, test_assembly_cpu_values)
        trial_assembly_gpu_values = move(backend, trial_assembly_cpu_values)
        
        if typeof(configuration["writeBackStrategy"]) == GpuWriteBackTrueInstance
            test_assembly_gpu_indexes = move(backend, test_assembly_cpu_indexes)
            trial_assembly_gpu_indexes = move(backend, trial_assembly_cpu_indexes)
        else
            test_assembly_gpu_indexes = 0
            trial_assembly_gpu_indexes = 0
        end

        
        

        GC.@preserve test_elements_vertices_matrix test_elements_tangents_matrix test_elements_volume_matrix trial_elements_vertices_matrix trial_elements_tangents_matrix trial_elements_volume_matrix test_assembly_gpu_values trial_assembly_gpu_values trial_assembly_gpu_indexes test_assembly_gpu_indexes begin

            elements_data = [test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix, trial_elements_vertices_matrix, trial_elements_tangents_matrix, trial_elements_volume_matrix]
            assembly_gpu_data = [test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, test_assembly_cpu_indexes, trial_assembly_cpu_indexes]
            
            time_quadrule_types += @elapsed begin   
                quadrule_types_gpu = KernelAbstractions.allocate(backend, Int8, elements_length_tuple)
                
                should_calc = create_results_matrix_gpu(backend, Int8, elements_length_tuple, ShouldCalcInstance)
                
                abs2_ = abs2(biop.gamma)
                floatmax_type = floatmax(type)

                # @code_warntype quadrule_determine_type(backend, 512)(quadrule_types_gpu, should_calc, abs2(biop.gamma), test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, type, ndrange = elements_length_tuple)
                
                quadrule_determine_type(backend, 512)(quadrule_types_gpu, should_calc, abs2_, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatmax_type, ShouldCalcInstance, ndrange = elements_length_tuple)
                KernelAbstractions.synchronize(backend)
            end






            SauterSchwabQuadratureCommonVertex = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonVertexCustomGpuData}()
            SauterSchwabQuadratureCommonEdge = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonEdgeCustomGpuData}()
            SauterSchwabQuadratureCommonFace = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonFaceCustomGpuData}()

            t = Threads.@spawn begin
                quadrule_types = Array{Int8}(undef, elements_length_tuple)
                KernelAbstractions.copyto!(CPU(), quadrule_types, quadrule_types_gpu)

                array = []
                threads_array = []
                nthreads = Threads.nthreads()
                indexes_ = [round(Int,s) for s in range(0, stop=test_elements_length_, length=nthreads+1)]
                for i in 1:nthreads
                    SauterSchwabQuadratureCommonVertex_ = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonVertexCustomGpuData}()
                    SauterSchwabQuadratureCommonEdge_ = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonEdgeCustomGpuData}()
                    SauterSchwabQuadratureCommonFace_ = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonFaceCustomGpuData}()
                    counts_ = zeros(3)
                    
                    push!(array, [[SauterSchwabQuadratureCommonVertex_, SauterSchwabQuadratureCommonEdge_, SauterSchwabQuadratureCommonFace_], counts_, [indexes_[i]+1, indexes_[i+1]]])
                end

                for i in 1:nthreads
                    entry = array[i]
                    job = Threads.@spawn sort_quadrule_into_custom_datastruct(entry[1][1], entry[1][2], entry[1][3], entry[2], entry[3][1], entry[3][2], pref_offet, trial_elements_length, test_elements, trial_elements, quadrule_types)
                    push!(threads_array, job)
                end

                for i in 1:nthreads
                    wait(threads_array[i])
                end
                
                SauterSchwabQuadratureCommonVertex = merge_into_first!([i[1][1] for i in array])
                SauterSchwabQuadratureCommonEdge = merge_into_first!([i[1][2] for i in array])
                SauterSchwabQuadratureCommonFace = merge_into_first!([i[1][3] for i in array])

                for i in 1:nthreads
                    counts[2] += array[i][2][1]
                    counts[3] += array[i][2][2]     
                    counts[4] += array[i][2][3]
                end

                counts[1] += test_elements_length_*trial_elements_length - (counts[2] + counts[3] + counts[4])
            end
            
            
            time_double_int += @elapsed begin
                wait(create_results)
                schedule_kernel!(
                    backend, 
                    length_return_matrix, elements_length_tuple,
                    assembly_gpu_data,
                    biop, should_calc, qd, type, store,
                    time_table, time_to_store, pref_offet,
                    elements_data,
                    configuration,
                    configuration["writeBackStrategy"]
                )
            end

            

            wait(t) 
            time_sauter_schwab += @elapsed begin
                sauterschwab_task_1 = Threads.@spawn sauterschwab_parameterized_gpu_outside_loop!(SauterSchwabQuadratureCommonVertex, SauterSchwabQuadrature.CommonVertex(qd.gausslegendre[1]), 
                    assembly_gpu_data, 
                    biop, SauterSchwabQuadratureCommonVertexCustomGpuDataInstance(), time_table, 2,
                    elements_data,
                    store, length_return_matrix, time_to_store, elements_length_tuple, configuration)

                sauterschwab_task_2 = Threads.@spawn sauterschwab_parameterized_gpu_outside_loop!(SauterSchwabQuadratureCommonEdge, SauterSchwabQuadrature.CommonEdge(qd.gausslegendre[2]),
                    assembly_gpu_data, 
                    biop, SauterSchwabQuadratureCommonEdgeCustomGpuDataInstance(), time_table, 3,
                    elements_data,
                    store, length_return_matrix, time_to_store, elements_length_tuple, configuration)

                sauterschwab_task_3 = Threads.@spawn sauterschwab_parameterized_gpu_outside_loop!(SauterSchwabQuadratureCommonFace, SauterSchwabQuadrature.CommonFace(qd.gausslegendre[3]),
                    assembly_gpu_data, 
                    biop, SauterSchwabQuadratureCommonFaceCustomGpuDataInstance(), time_table, 4,
                    elements_data,
                    store, length_return_matrix, time_to_store, elements_length_tuple, configuration)
                    
                wait(sauterschwab_task_1)
                wait(sauterschwab_task_2)
                wait(sauterschwab_task_3)
            end

            
            pref_offet = offset

        end
    end end

    # @show time_overhead

    # if isdefined(Main, :time_logger)
    #     function log_time(time_logger, key::String, value)
    #         if haskey(time_logger, key)
    #             push!(time_logger[key], value)
    #         else
    #             time_logger[key] = [value]
    #         end
    #     end
    #     log_time(time_logger, "time overhead", time_overhead)
    #     log_time(time_logger, "time to determin the quadrule", time_quadrule_types)
    #     log_time(time_logger, "calculate the double int", time_double_int)
    #     log_time(time_logger, "transfer quadrule to CPU", time_transfer_to_CPU)
    #     log_time(time_logger, "calculate double for loop", time_double_forloop)
    #     log_time(time_logger, "calculate SauterSchwab", time_sauter_schwab)
    #     for i in 2:4
    #         log_time(time_logger, "time_sauter_schwab_overhead_and_test_toll $i", time_table[1,i])
    #         log_time(time_logger, "calc_sauter_schwab $i", time_table[2,i])
    #     end
    #     log_time(time_logger, "time_table[1,:]", time_table[1,:])
    #     log_time(time_logger, "time_table[2,:]", time_table[2,:])
    #     log_time(time_logger, "time_to_store", time_to_store)
    # end
end



function move(backend, input)
    out = KernelAbstractions.allocate(backend, eltype(input), size(input))
    # @noswitch begin
        KernelAbstractions.copyto!(backend, out, input)
    # end
    return out
end

function move(backend, out, input)
    KernelAbstractions.copyto!(backend, out, input)
end
