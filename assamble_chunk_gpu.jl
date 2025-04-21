# include("../src/quadrature/doublenumints.jl")
# using .Zzz      

include("doubleQuadRule_3d_gpu_outside_loops_and_store.jl")
# using .DoubleQuadRule_3d_gpu_outside_loops_and_store
using Serialization

include("CustomDataStructs/SauterSchwabQuadratureDataStruct.jl")
include("CustomDataStructs/doubleQuadRuleGpuStrategy.jl")

include("GPU_scheduler.jl")

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

function assemblechunk_gpu!(biop::BEAST.IntegralOperator, tfs::BEAST.Space, bfs::BEAST.Space, writeBackStrategy::GpuWriteBack, amount_of_gpus, store;
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
        qd, zlocal, writeBackStrategy, amount_of_gpus, store; quadstrat=qs)
end

function validate_and_extract(data, size_qrule)
    size_ = (1,3,size_qrule)
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

function load_data_to_custum_data_for_gpu!(tcell, bcell, qrule_and_datastorages, p, q)
    qrule = qrule_and_datastorages[1]
    cpu_data = qrule_and_datastorages[2]

    add_element(cpu_data, SVector{4}(map(t -> SVector{2}(t...), qrule.qps)), SVector{2, Int64}(p, q)) 
end

const dtol = 2.220446049250313e-16 * 1.0e3
const xtol2 = 0.2 * 0.2
@kernel function quadrule_determine_type(result::AbstractArray, should_calc_gpu_::AbstractArray, @Const(k2::Float64), @Const(τ::AbstractArray), @Const(σ::AbstractArray), @Const(σ_volume::AbstractArray), @Const(floatmax_type::Float64))
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


function transformHitsToSauterSchwabQuadrature(hits, qd , SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
    hits == 1 && return (SauterSchwabQuadrature.CommonVertex(qd.gausslegendre[hits]), SauterSchwabQuadratureCommonVertex)
    hits == 2 && return (SauterSchwabQuadrature.CommonEdge(qd.gausslegendre[hits]), SauterSchwabQuadratureCommonEdge)
    hits == 3 && return (SauterSchwabQuadrature.CommonFace(qd.gausslegendre[hits]), SauterSchwabQuadratureCommonFace)
end

function load_data(backend, type, size_qrule, test_elements)
    elements_vertices_matrix = Array{type}(undef, size_qrule, 3, 3)
    elements_tangents_matrix = Array{type}(undef, size_qrule, 3, 3)
    elements_volume_matrix = Array{type}(undef, size_qrule)
    for p in 1:size_qrule
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
    pref_offet, trial_elements_length, test_elements, trial_elements, quadrule_types, qd)

    # @show SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace
    # @show counts
    # @show start, stop


    for p in start:stop
        tcell = test_elements[p + pref_offet]
        for q in 1:trial_elements_length
            if quadrule_types[p,q] != 0
                # time = @elapsed begin 
                    # index = ((p - 1) ÷ 32 ) + 1
                    # bitindex = ((p - 1) % 32 ) + 1
                    # should_calc[index,q] = should_calc[index, q] & ~(1 << (bitindex - 1))

                    # should_calc[p,q] = 0

                bcell = trial_elements[q]
                qrule_and_datastorages = transformHitsToSauterSchwabQuadrature(quadrule_types[p,q], qd, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
                
                load_data_to_custum_data_for_gpu!(tcell, bcell, qrule_and_datastorages, p, q)
                # end
                # time_table[1, quadrule_types[p,q] + 1] += time
                # if quadrule_types[p,q] != 1 && quadrule_types[p,q] != 2 && quadrule_types[p,q] != 3
                #     @show quadrule_types[p,q]
                # end
                # counts[quadrule_types[p,q]] += 1
            end
        end
    end
end




function assemblechunk_body_gpu!(biop,
    test_space, test_elements, test_assembly_data, test_cell_ptrs,
    trial_space, trial_elements, trial_assembly_data, trial_cell_ptrs,
    qd, zlocal, writeBackStrategy::GpuWriteBack, amount_of_gpus, store; quadstrat)
    type = Float64

    test_elements_length = length(test_elements)
    trial_elements_length = length(trial_elements)


    GPU_budget = 3 * 2^30
    # test_sub_elements_length = x
    # trail_sub_elements_length = y
    # length_CommonVertex = z1
    # length_CommonEdge = z2
    # length_CommonFace = z3
    # calc_budget =   3 * test_sub_elements_length * size(Float64)                                                    # size(test_assembly_gpu_values) 
    #                 + 3 * trail_sub_elements_length * size(Float64)                                                 # size(trial_assembly_gpu_values) 
    #                 + 3 * test_sub_elements_length * size(Float64)                                                  # size(test_assembly_gpu_indexes) 
    #                 + 3 * trail_sub_elements_length * size(Float64)                                                 # size(trial_assembly_gpu_indexes)

    #                 + test_sub_elements_length *  trail_sub_elements_length * size(Int8)                            # size(quadrule_types_gpu)
    #                 + test_sub_elements_length *  trail_sub_elements_length * size(Int8)                            # size(should_calc)
    #                 + (3 * test_sub_elements_length + 2 * 3 * 3 * test_sub_elements_length                          # size(womps_weights, womps_values, womps_cart)
    #                 + 4 * trail_sub_elements_length + 2 * 3 * 4 * trail_sub_elements_length) * sizeof(type)         # size(wimps_weights, wimps_values, wimps_cart)
    #                 + 1 * 2^30                                                                                     # size of return matrix on gpu for dubble Int

                    
    #                 +   ((3 * 2) * size(Float64)                                                                    # size(ichart1_vert) 
    #                 +    (3 * 2) * size(Float64)                                                                    # size(ichart2_vert) 
    #                 +    (2 * 2) * size(Float64)                                                                    # size(ichart1_tan) 
    #                 +    (2 * 2) * size(Float64)                                                                    # size(ichart2_tan) 
    #                 +    (4 * 2) * size(Float64)                                                                    #size(qps)
    #                 +    (2) * size(Int64)                                                                          #size(store_index)
    #                 ) * (z1 + z2 + z3)                                                                              # for Vertex,z Edge and Face
    #                 +    1 * 2^30 * 3                                                                              # size of return matrix on gpu for dubble Int
        


    
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

    # @show GPU_budget
    # @show calc_budget

    if GPU_budget >= calc_budget
        range_test = 1:amount_of_gpus
        range_trail = 1:1
    else
        # @show sizeof(test_elements_length), sizeof(trial_elements_length)
        # submatrix_size, n_rows, n_cols = choose_square_submatrix_shape(test_elements_length, trial_elements_length, GPU_budget)
        # @show submatrix_size, n_rows, n_cols
        # submatrix_size, n_rows, n_cols = choose_square_submatrix_shape(calc_budget, GPU_budget)
        # @show submatrix_size, n_rows, n_cols
        range_test = 1:1
        range_trail = 1:1
        # return
    end


                    
    # TOT_gpu_buget =   3 * test_elements_length * size(Float64)                                                # size(test_assembly_gpu_values) 
    #             + 3 * trial_elements_length * size(Float64)                                                 # size(trial_assembly_gpu_values) 
    #             + 3 * test_elements_length * size(Float64)                                                  # size(test_assembly_gpu_indexes) 
    #             + 3 * trial_elements_length * size(Float64)                                                 # size(trial_assembly_gpu_indexes)
    #             + test_elements_length *  trial_elements_length * size(Int8)                            # size(quadrule_types_gpu)
    #             + test_elements_length *  trial_elements_length * size(Int8)                            # size(should_calc)
    #             + (3 * test_elements_length + 2 * 3 * 3 * test_elements_length                          # size(womps_weights, womps_values, womps_cart)
    #             + 4 * trial_elements_length + 2 * 3 * 4 * trial_elements_length) * sizeof(type)         # size(wimps_weights, wimps_values, wimps_cart)
    #             + 1 * 10^30                                                                                     # size of return matrix on gpu for dubble Int

                
    #             +   ((length * 3 * 2) * size(Float64)                                                           # size(ichart1_vert) 
    #             +    (length * 3 * 2) * size(Float64)                                                           # size(ichart2_vert) 
    #             +    (length * 2 * 2) * size(Float64)                                                           # size(ichart1_tan) 
    #             +    (length * 2 * 2) * size(Float64)                                                           # size(ichart2_tan) 
    #             +    (length* 4 * 2) * size(Float64)                                                            #size(qps)
    #             +    (length * 2) * size(Int64)                                                                 #size(store_index)
    #             ) * (z1 + z2 + z3)                                                                              # for Vertex,z Edge and Face
    #             +    1 * 10^30 * 3                                                                              # size of return matrix on gpu for dubble Int

    # _OffsetStore(store, lo-1, 0)





    # time_overhead = @elapsed begin
        empty!(gpu_results_cache)


        test_shapes = refspace(test_space)
        trial_shapes = refspace(trial_space)

        myid = Threads.threadid()
        # myid == 1 && print("dots out of 10: ")
        todo, done, pctg = length(test_elements), 0, 0
        length_return_matrix = length(test_space.geo.vertices)
        size_qrule = todo

        indexes = [round(Int,s) for s in range(0, stop=length(test_elements), length=amount_of_gpus+1)]
        lengths = ones(Int, amount_of_gpus)
        for i in 1:amount_of_gpus
            lengths[i] = indexes[i+1] - indexes[i]
        end

        
        counts = zeros(4)
        time_table = [Atomic{Float64}(0.0) for _ in 1:2, _ in 1:4]
        
        SauterSchwabQuadratureCommonVertex = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonVertexCustomGpuData}()
        SauterSchwabQuadratureCommonEdge = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonEdgeCustomGpuData}()
        SauterSchwabQuadratureCommonFace = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonFaceCustomGpuData}()
        
        InstancedoubleQuadRuleGpuStrategyShouldCalculate = doubleQuadRuleGpuStrategyShouldCalculateInstance()
        
        
        test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix = load_data(backend, type, test_elements_length, test_elements)
        trial_elements_vertices_matrix, trial_elements_tangents_matrix, trial_elements_volume_matrix = load_data(backend, type, trial_elements_length, trial_elements)
        trial_elements_vertices_matrix, trial_elements_tangents_matrix, trial_elements_volume_matrix = move(backend, trial_elements_vertices_matrix), move(backend, trial_elements_tangents_matrix), move(backend, trial_elements_volume_matrix)
    

        SauterSchwabQuadratureCommonVertexCustomGpuDataInstance_ = SauterSchwabQuadratureCommonVertexCustomGpuDataInstance()
        SauterSchwabQuadratureCommonEdgeCustomGpuDataInstance_ = SauterSchwabQuadratureCommonEdgeCustomGpuDataInstance()
        SauterSchwabQuadratureCommonFaceCustomGpuDataInstance_ = SauterSchwabQuadratureCommonFaceCustomGpuDataInstance()


        test_assembly_cpu_values, test_assembly_cpu_indexes = validate_and_extract(test_assembly_data, test_elements_length)
        trial_assembly_cpu_values, trial_assembly_cpu_indexes = validate_and_extract(trial_assembly_data, trial_elements_length)

        time_to_store = Threads.Atomic{Float64}(0)
    # end
    # @show time_overhead

    
    test_elements_vertices_matrix_original, test_elements_tangents_matrix_original, test_elements_volume_matrix_original = test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix
    trial_elements_vertices_matrix_original, trial_elements_tangents_matrix_original, trial_elements_volume_matrix_original = trial_elements_vertices_matrix, trial_elements_tangents_matrix, trial_elements_volume_matrix

    test_assembly_cpu_values_original, test_assembly_cpu_indexes_original = test_assembly_cpu_values, test_assembly_cpu_indexes
    trial_assembly_cpu_values_original, trial_assembly_cpu_indexes_original = trial_assembly_cpu_values, trial_assembly_cpu_indexes
    
    
    offset = 0
    pref_offet = 0

    time_double_int = 0

    for i in range_test; for j in range_trail

    
    elements_length_tuple = (test_elements_length, trial_elements_length)

    

    i_start, i_end = indexes[i]+1, indexes[i+1]
    test_elements_length_ = lengths[i]
    offset += test_elements_length_

    @show test_elements_length_ * trial_elements_length
    continue  
    
    elements_length_tuple = (test_elements_length_, trial_elements_length)

    test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix = test_elements_vertices_matrix_original[i_start:i_end,:,:], test_elements_tangents_matrix_original[i_start:i_end,:,:], test_elements_volume_matrix_original[i_start:i_end]
    test_assembly_cpu_values, test_assembly_cpu_indexes = test_assembly_cpu_values_original[:,i_start:i_end], test_assembly_cpu_indexes_original[:,i_start:i_end]

    test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix = move(backend, test_elements_vertices_matrix), move(backend, test_elements_tangents_matrix), move(backend, test_elements_volume_matrix)
    

    reset!(SauterSchwabQuadratureCommonVertex)
    reset!(SauterSchwabQuadratureCommonEdge)
    reset!(SauterSchwabQuadratureCommonFace)

    test_assembly_gpu_values = move(backend, test_assembly_cpu_values)
    trial_assembly_gpu_values = move(backend, trial_assembly_cpu_values)
    
    if typeof(writeBackStrategy) == GpuWriteBackTrueInstance
        test_assembly_gpu_indexes = move(backend, test_assembly_cpu_indexes)
        trial_assembly_gpu_indexes = move(backend, trial_assembly_cpu_indexes)
    else
        test_assembly_gpu_indexes = 0
        trial_assembly_gpu_indexes = 0
    end

    time_quadrule_types = @elapsed begin   
        quadrule_types_gpu = KernelAbstractions.allocate(backend, Int8, elements_length_tuple)
        # should_calc_gpu_ = KernelAbstractions.allocate(backend, UInt32, (ceil(Int, size_qrule / 32) * 32, size_qrule))
        # should_calc_gpu_ = KernelAbstractions.allocate(backend, UInt32, (ceil(Int, size_qrule / 32) * 32, size_qrule))
        should_calc = KernelAbstractions.allocate(backend, Int8, elements_length_tuple)
        
        abs2_ = abs2(biop.gamma)
        floatmax_type = floatmax(type)

        # @code_warntype quadrule_determine_type(backend, 512)(quadrule_types_gpu, should_calc, abs2(biop.gamma), test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, type, ndrange = elements_length_tuple)
        quadrule_determine_type(backend, 512)(quadrule_types_gpu, should_calc, abs2_, test_elements_vertices_matrix, trial_elements_vertices_matrix, trial_elements_volume_matrix, floatmax_type, ndrange = elements_length_tuple)
        KernelAbstractions.synchronize(backend)
        quadrule_types = Array(quadrule_types_gpu)
        # @show quadrule_types
        @assert !(4 in quadrule_types)
    end
    @show time_quadrule_types


   

    t = Threads.@spawn begin
        # time_double_forloop = @elapsed begin
            array = []
            threads_array = []
            nthreads = Threads.nthreads()
            indexes_ = [round(Int,s) for s in range(0, stop=test_elements_length_, length=nthreads+1)]
            for i in 1:nthreads
                SauterSchwabQuadratureCommonVertex_ = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonVertexCustomGpuData}()
                SauterSchwabQuadratureCommonEdge_ = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonEdgeCustomGpuData}()
                SauterSchwabQuadratureCommonFace_ = SauterSchwabQuadrature_gpu_data{SauterSchwabQuadratureCommonFaceCustomGpuData}()
                counts_ = zeros(3)
                # @show counts_
                
                push!(array, [[SauterSchwabQuadratureCommonVertex_, SauterSchwabQuadratureCommonEdge_, SauterSchwabQuadratureCommonFace_], counts_, [indexes_[i]+1, indexes_[i+1]]])
            end

            for i in 1:nthreads
                # SauterSchwabQuadratureCommonVertex = array[1][1]
                # SauterSchwabQuadratureCommonEdge = array[1][2]
                # SauterSchwabQuadratureCommonFace = array[1][3]
                # counts = array[2]
                # start, stop = array[3][1], array[3][2]
                entry = array[i]
                job = Threads.@spawn sort_quadrule_into_custom_datastruct(entry[1][1], entry[1][2], entry[1][3], entry[2], entry[3][1], entry[3][2], pref_offet, trial_elements_length, test_elements, trial_elements, quadrule_types, qd)
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
        # end
        # @show time_double_forloop

        # time_double_forloop = @elapsed begin
        #     for p in 1:test_elements_length_
        #         tcell = test_elements[p + pref_offet]
        #         for q in 1:trial_elements_length
        #             if quadrule_types[p,q] != 0
        #                 # time = @elapsed begin 
        #                     # index = ((p - 1) ÷ 32 ) + 1
        #                     # bitindex = ((p - 1) % 32 ) + 1
        #                     # should_calc[index,q] = should_calc[index, q] & ~(1 << (bitindex - 1))

        #                     # should_calc[p,q] = 0

        #                     bcell = trial_elements[q]
        #                     qrule_and_datastorages = transformHitsToSauterSchwabQuadrature(quadrule_types[p,q], qd, SauterSchwabQuadratureCommonVertex, SauterSchwabQuadratureCommonEdge, SauterSchwabQuadratureCommonFace)
                    
        #                     load_data_to_custum_data_for_gpu!(tcell, bcell, qrule_and_datastorages, p, q)
        #                     # end
        #                 # time_table[1, quadrule_types[p,q] + 1] += time
        #             end
        #             counts[quadrule_types[p,q] + 1] += 1
        #         end
        #     end
        # end
        # @show time_double_forloop

        sauterschwab_task_1 = Threads.@spawn sauterschwab_parameterized_gpu_outside_loop!(SauterSchwabQuadratureCommonVertex, 
            test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, 
            biop, SauterSchwabQuadratureCommonVertexCustomGpuDataInstance_, writeBackStrategy, time_table, 2,
            test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix,
            trial_elements_vertices_matrix, trial_elements_tangents_matrix, trial_elements_volume_matrix,
            store, length_return_matrix, size_qrule, SauterSchwabQuadratureCommonVertex, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, time_to_store, elements_length_tuple)

        sauterschwab_task_2 = Threads.@spawn sauterschwab_parameterized_gpu_outside_loop!(SauterSchwabQuadratureCommonEdge, 
            test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, 
            biop, SauterSchwabQuadratureCommonEdgeCustomGpuDataInstance_, writeBackStrategy, time_table, 3,
            test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix,
            trial_elements_vertices_matrix, trial_elements_tangents_matrix, trial_elements_volume_matrix,
            store, length_return_matrix, size_qrule, SauterSchwabQuadratureCommonEdge, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, time_to_store, elements_length_tuple)

        sauterschwab_task_3 = Threads.@spawn sauterschwab_parameterized_gpu_outside_loop!(SauterSchwabQuadratureCommonFace, 
            test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, 
            biop, SauterSchwabQuadratureCommonFaceCustomGpuDataInstance_, writeBackStrategy, time_table, 4,
            test_elements_vertices_matrix, test_elements_tangents_matrix, test_elements_volume_matrix,
            trial_elements_vertices_matrix, trial_elements_tangents_matrix, trial_elements_volume_matrix,
            store, length_return_matrix, size_qrule, SauterSchwabQuadratureCommonFace, test_assembly_cpu_indexes, trial_assembly_cpu_indexes, time_to_store, elements_length_tuple)
            
        wait(sauterschwab_task_1)
        wait(sauterschwab_task_2)
        wait(sauterschwab_task_3)
    end
    

    time_double_int += @elapsed begin
        sk = Threads.@spawn schedule_kernel!(
            backend, 
            length_return_matrix, size_qrule, elements_length_tuple,
            writeBackStrategy, InstancedoubleQuadRuleGpuStrategyShouldCalculate,
            test_assembly_gpu_indexes, trial_assembly_gpu_indexes, test_assembly_gpu_values, trial_assembly_gpu_values, test_assembly_cpu_indexes, trial_assembly_cpu_indexes,
            biop, should_calc, qd, type, store,
            time_table, time_to_store, pref_offet
            # producers
        )
        wait(sk)
    end

    # Threads.atomic_add!(time_table[2,1], time_double_int)
    # time_table[2,1] = time_double_int


    

    wait(t) 

    
    pref_offet = offset

    end end

    @show time_double_int


    # @show time_table[1,:]
    # @show time_table[2,:]
    # @show counts
    # @show time_to_store

end


function move(backend, input)
    out = KernelAbstractions.allocate(backend, eltype(input), size(input))
    return KernelAbstractions.copyto!(backend, out, input)
end
