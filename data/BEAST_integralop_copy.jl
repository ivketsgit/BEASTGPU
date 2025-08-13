include("../utils/benchmark_alternative.jl")

abstract type AbstractOperator end
abstract type Operator <: AbstractOperator end
abstract type IntegralOperator <: Operator end
abstract type MaxwellOperator3D{T,K} <: IntegralOperator end
abstract type Helmholtz3DOp{T,K} <: MaxwellOperator3D{T,K} end
# abstract type Helmholtz3DOp{T,K} <: BEAST.MaxwellOperator3D{T,K} end

using SauterSchwabQuadrature
    
const    PRINT_LOCK = ReentrantLock()

struct HH3DSingleLayerFDBIO{T,K} <: Helmholtz3DOp{T,K}
    alpha::T
    gamma::K
end

function assemblechunk_gpu!(biop::BEAST.IntegralOperator, tfs::BEAST.Space, bfs::BEAST.Space, store,print_file,appendOrWrite;
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
        qd, zlocal, store,print_file,appendOrWrite; quadstrat=qs)
end




function assemblechunk_body_gpu!(biop,
    test_space, test_elements, test_assembly_data, test_cell_ptrs,
    trial_space, trial_elements, trial_assembly_data, trial_cell_ptrs,
    qd, zlocal, store,print_file,appendOrWrite; quadstrat)

    
    
    #BEAST.jl loads
    test_shapes = refspace(test_space)
    trial_shapes = refspace(trial_space)

    todo, done, pctg = length(test_elements), 0, 0
    length_return_matrix = length(test_space.geo.vertices)

    
    # f = function()
    #     for (p, tcell) in enumerate(test_elements)
    #         for (q, bcell) in enumerate(trial_elements)
    #             fill!(zlocal, 0)
    #         end
    #     end
    # end


    # times = manual_benchmark(f, n=100,filename= print_file * "/fill.txt",  appendOrWrite=appendOrWrite)
    

    # f = function()
    #     for (p, tcell) in enumerate(test_elements)
    #         for (q, bcell) in enumerate(trial_elements)
    #             BEAST.quadrule(biop, test_space, trial_space, p, tcell, q, bcell, qd, quadstrat)
    #         end
    #     end
    # end

    
    # times = manual_benchmark(f, n=100,filename= print_file * "/quadrule.txt",  appendOrWrite=appendOrWrite)

    times = [0.0,0.0,0.0,0.0]
    for (p, tcell) in enumerate(test_elements)
        for (q, bcell) in enumerate(trial_elements)
            qrule = BEAST.quadrule(biop, test_space, trial_space, p, tcell, q, bcell, qd, quadstrat)
            index = 0
            if  typeof(qrule) ==  SauterSchwabQuadrature.CommonVertex{Vector{Tuple{Float64, Float64}}}
                index = 2
            elseif  typeof(qrule) ==  SauterSchwabQuadrature.CommonEdge{Vector{Tuple{Float64, Float64}}}
                index = 3
            elseif  typeof(qrule) ==  SauterSchwabQuadrature.CommonFace{Vector{Tuple{Float64, Float64}}}
                index = 4
            else
                index = 1
            end
            time_ = @elapsed begin
                BEAST.momintegrals!(biop, test_shapes, trial_shapes, tcell, bcell, zlocal, qrule)
            end
            times[index] += time_
            # BEAST.momintegrals!(biop, test_shapes, trial_shapes, tcell, bcell, zlocal, qrule)
        end
    end
    # println(times)  
    lock(PRINT_LOCK) do
        
        open(print_file * "/momintegrals_main.txt",  appendOrWrite) do file
            print(file,times[1])
            print(file,",")
        end
        open(print_file * "/momintegrals_non_main_V.txt",  appendOrWrite) do file
            print(file,times[2])
            print(file,",")
        end
        open(print_file * "/momintegrals_non_main_E.txt",  appendOrWrite) do file
            print(file,times[3])
            print(file,",")
        end
        open(print_file * "/momintegrals_non_main_F.txt",  appendOrWrite) do file
            print(file,times[4])
            print(file,",")
        end
    end
    # f = function()
    #     for (p, tcell) in enumerate(test_elements)
    #         for (q, bcell) in enumerate(trial_elements)
    #             qrule = BEAST.quadrule(biop, test_space, trial_space, p, tcell, q, bcell, qd, quadstrat)
    #             if  typeof(qrule) ==  SauterSchwabQuadrature.CommonVertex{Vector{Tuple{Float64, Float64}}}
    #                 BEAST.momintegrals!(biop, test_shapes, trial_shapes, tcell, bcell, zlocal, qrule)
    #             end
    #         end
    #     end
    # end
    # times = manual_benchmark(f, n=10,filename= print_file * "/momintegrals_non_main_1.txt", max_hours=0.2,  appendOrWrite=appendOrWrite)
    
    # f = function()
    #     for (p, tcell) in enumerate(test_elements)
    #         for (q, bcell) in enumerate(trial_elements)
    #             qrule = BEAST.quadrule(biop, test_space, trial_space, p, tcell, q, bcell, qd, quadstrat)
    #             if  typeof(qrule) ==  SauterSchwabQuadrature.CommonEdge{Vector{Tuple{Float64, Float64}}}
    #                 BEAST.momintegrals!(biop, test_shapes, trial_shapes, tcell, bcell, zlocal, qrule)
    #             end
    #         end
    #     end
    # end
    # times = manual_benchmark(f, n=10,filename= print_file * "/momintegrals_non_main_2.txt", max_hours=0.2,  appendOrWrite=appendOrWrite)
    
    # f = function()
    #     for (p, tcell) in enumerate(test_elements)
    #         for (q, bcell) in enumerate(trial_elements)
    #             qrule = BEAST.quadrule(biop, test_space, trial_space, p, tcell, q, bcell, qd, quadstrat)
    #             if  typeof(qrule) ==  SauterSchwabQuadrature.CommonFace{Vector{Tuple{Float64, Float64}}}
    #                 BEAST.momintegrals!(biop, test_shapes, trial_shapes, tcell, bcell, zlocal, qrule)
    #             end
    #         end
    #     end
    # end
    # times = manual_benchmark(f, n=10,filename= print_file * "/momintegrals_non_main_3.txt", max_hours=0.2, appendOrWrite=appendOrWrite)
    
    

    # fill!(zlocal, 0)
    # f = function()
    #     for (p, tcell) in enumerate(test_elements)
    #         for (q, bcell) in enumerate(trial_elements)
    #             I = length(test_assembly_data[p])
    #             J = length(test_assembly_data[q])
    #             for j in 1 : J, i in 1 : I
    #                 zij = zlocal[i,j]
    #                 for  (n,b) in trial_assembly_data[q][j]
    #                     zb =  zij*b
    #                     for (m, a) in test_assembly_data[p][i]
    #                         store(a*zb, m, n)
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end

    
    # times = manual_benchmark(f, n=100,filename=print_file * "/store.txt", appendOrWrite=appendOrWrite)
    

    
    
    
end
