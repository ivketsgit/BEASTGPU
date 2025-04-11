using BEAST
using CompScienceMeshes
using BenchmarkTools
using Serialization




include("assamble_gpu.jl")
include("utils/backend.jl")

@kernel function warmup_gpu_kernel()
    # i = @index(Global, Linear)
end

function warmup_gpu()
    time_warmup = @elapsed begin
        # KernelAbstractions.allocate(backend, Int8, 256)
        warmup_gpu_kernel(backend)(ndrange = 1)
    end
    @show time_warmup
end
Threads.@spawn warmup_gpu()

warmup_thread = Threads.@spawn begin
    writeBackStrategy = GpuWriteBackFalseInstance()
    SauterSchwabQuadratureCommonVertexCustomGpuDataInstance_ = SauterSchwabQuadratureCommonVertexCustomGpuDataInstance()
    SauterSchwabQuadratureCommonEdgeCustomGpuDataInstance_ = SauterSchwabQuadratureCommonEdgeCustomGpuDataInstance()
    SauterSchwabQuadratureCommonFaceCustomGpuDataInstance_ = SauterSchwabQuadratureCommonFaceCustomGpuDataInstance()
    γ = Float64(1.0)
    α = Float64(1.0)
    

    warmup_result = KernelAbstractions.zeros(backend, ComplexF64, (1,9))
    warmup_qps = KernelAbstractions.ones(backend, Float64, (1,4,2))

    warmup_test_vert = KernelAbstractions.zeros(backend, Float64, (1,3,3))
    warmup_trail_vert = KernelAbstractions.ones(backend, Float64, (1,3,3))
    warmup_test_tan = KernelAbstractions.zeros(backend, Float64, (1,3,3))
    warmup_trail_tan = KernelAbstractions.ones(backend, Float64, (1,3,3))
    warmup_test_vol = KernelAbstractions.zeros(backend, Float64, (1,3))
    warmup_trail_vol = KernelAbstractions.ones(backend, Float64, (1,3))

    warmup_ichart1_vert = KernelAbstractions.zeros(backend, Float64, (1,3,2))
    warmup_ichart2_vert = KernelAbstractions.ones(backend, Float64, (1,3,2))
    warmup_ichart1_tan = KernelAbstractions.zeros(backend, Float64, (1,2,2))
    warmup_ichart2_tan = KernelAbstractions.ones(backend, Float64, (1,2,2))

    warmup_store_index = KernelAbstractions.ones(backend, Int64, (1,2))

    warmup_test_assembly_gpu_indexes = KernelAbstractions.ones(backend, Int64, ())
    warmup_trial_assembly_gpu_indexes = KernelAbstractions.ones(backend, Int64, ())
    warmup_test_assembly_gpu_values = KernelAbstractions.ones(backend, Float64, (3,1))
    warmup_trial_assembly_gpu_values = KernelAbstractions.ones(backend, Float64, (3,1))

    
    warmup_womps_weights = KernelAbstractions.zeros(backend, Float64, (1,3))
    warmup_wimps_weights = KernelAbstractions.ones(backend, Float64, (1,4))
    warmup_womps_values = KernelAbstractions.zeros(backend, Float64, (1,3,3))
    warmup_wimps_values = KernelAbstractions.ones(backend, Float64, (1,3,4))
    warmup_womps_cart = KernelAbstractions.zeros(backend, Float64, (1,3,3))
    warmup_wimps_cart = KernelAbstractions.ones(backend, Float64, (1,3,4))

    warmup_should_calc = KernelAbstractions.ones(backend, Int8, (1,1))

    combined_kernel_temp_outside_loops_linear_index!(backend)(
        warmup_result,
        warmup_test_assembly_gpu_indexes, warmup_trial_assembly_gpu_indexes, warmup_test_assembly_gpu_values, warmup_trial_assembly_gpu_values, 
        1,
        γ, α,
        warmup_womps_weights, warmup_wimps_weights, 
        warmup_womps_values, warmup_wimps_values,
        warmup_womps_cart, warmup_wimps_cart, 
        0, 0,
        instance, writeBackStrategy,
        warmup_should_calc,
        warmup_store_index,
        ndrange = (1,1))
    
    sauterschwab_parameterized_gpu_outside_loop_kernel!(backend, 256)(warmup_result, warmup_qps, 
    warmup_test_vert, warmup_trail_vert, warmup_test_tan, warmup_trail_tan, warmup_test_vol, warmup_trail_vol, warmup_ichart1_vert, warmup_ichart2_vert, warmup_ichart1_tan, warmup_ichart2_tan, warmup_store_index, 
    warmup_test_assembly_gpu_indexes, warmup_trial_assembly_gpu_indexes, warmup_test_assembly_gpu_values, warmup_trial_assembly_gpu_values, 
    γ, α, SauterSchwabQuadratureCommonVertexCustomGpuDataInstance_, writeBackStrategy, ndrange = (1))
    sauterschwab_parameterized_gpu_outside_loop_kernel!(backend, 256)(warmup_result, warmup_qps, 
    warmup_test_vert, warmup_trail_vert, warmup_test_tan, warmup_trail_tan, warmup_test_vol, warmup_trail_vol, warmup_ichart1_vert, warmup_ichart2_vert, warmup_ichart1_tan, warmup_ichart2_tan, warmup_store_index, 
    warmup_test_assembly_gpu_indexes, warmup_trial_assembly_gpu_indexes, warmup_test_assembly_gpu_values, warmup_trial_assembly_gpu_values, 
    γ, α, SauterSchwabQuadratureCommonEdgeCustomGpuDataInstance_, writeBackStrategy, ndrange = (1))
    sauterschwab_parameterized_gpu_outside_loop_kernel!(backend, 256)(warmup_result, warmup_qps, 
    warmup_test_vert, warmup_trail_vert, warmup_test_tan, warmup_trail_tan, warmup_test_vol, warmup_trail_vol, warmup_ichart1_vert, warmup_ichart2_vert, warmup_ichart1_tan, warmup_ichart2_tan, warmup_store_index, 
    warmup_test_assembly_gpu_indexes, warmup_trial_assembly_gpu_indexes, warmup_test_assembly_gpu_values, warmup_trial_assembly_gpu_values, 
    γ, α, SauterSchwabQuadratureCommonFaceCustomGpuDataInstance_, writeBackStrategy, ndrange = (1))
end

inv_density_factor = 20
Γ = meshcuboid(1.0,1.0,1.0,0.5/inv_density_factor)
X = lagrangec0d1(Γ) 
S = Helmholtz3D.singlelayer(wavenumber = 1.0)
filename = "zzz/cashed_results/matrix_ref_$inv_density_factor.bin"


let time = @elapsed begin
    # @show @which assemble(S,X,X)
        M_ref = assemble(S,X,X)
    end
    # open(filename, "w") do io
    #     serialize(io, M_ref)
    # end
    println("Elapsed time control: ", time)
    println("")
end

let time = @elapsed begin
        M = assemble_gpu(S,X,X)
    end 
    println("Elapsed time: ", time)
    println("")
end
# @show M_ref
# @show m

# M_ref = open(filename, "r") do io
#     deserialize(io)
# end

error_matrix = abs.(M_ref .- M)
println("")
@show maximum(error_matrix)
# @show M
# @show M_ref
# for (i, e) in enumerate(abs.(M_ref .- M))
#     @show i, e
# end

# function test_functions(min, steps, max, iterations)
#     dir = "zzz\\data"
#     groups = Dict(
#         "control" => ["CommonVertex_control","CommonEdge_control","CommonFace_control","DoubleQuadRule_control"],
#         "load" => ["DoubleQuadRule_load","repaire_load","CommonVertex_load","CommonEdge_load","CommonFace_load"],
#         "calc" => ["DoubleQuadRule_calc","repaire_calc","CommonVertex_calc","CommonEdge_calc","CommonFace_calc"]
#     )
#     names_control = ["CommonVertex_control","CommonEdge_control","CommonFace_control","DoubleQuadRule_control"]
#     names_load = ["DoubleQuadRule_load","repaire_load","CommonVertex_load","CommonEdge_load","CommonFace_load"]
#     names_calc = ["DoubleQuadRule_calc","repaire_calc","CommonVertex_calc","CommonEdge_calc","CommonFace_calc"]
    
#     Γ = Dict()
#     for i in min:steps:max
#         Γ[i] = meshcuboid(1.0,1.0,1.0,0.5/i)
#     end

    
#     X = Dict()
#     for i in min:steps:max
#         X[i] = lagrangec0d1(Γ[i]) 
#     end
#     S = Helmholtz3D.singlelayer(wavenumber = 1.0)


#     for iteration in 1:iterations
#         for group in values(groups)
#             for name in group
#                 open(joinpath(dir, name), "a") do io
#                     write(io, "[")
#                 end
#             end
#         end


#         println("CONTROL")
#         for i in min:steps:max
#             println(i)
#             @benchmark M_ref = assemble($S,$(X[i]),$(X[i])) samples=1 evals=1
#             for name in names_control
#                 if max - i >= steps
#                     open(joinpath(dir, name), "a") do io
#                         write(io, ",")
#                     end
#                 end
#             end
#         end

#         println("\n")
#         println("EXPERIMENT")
#         for i in min:steps:max
#             println(i)
#             @benchmark M = assemble_gpu($S,$(X[i]),$(X[i])) samples=1 evals=1
#             if max - i >= steps
#                 for name in names_load
#                     open(joinpath(dir, name), "a") do io
#                         write(io, ",")
#                     end
#                 end
            
#                 for name in names_calc
#                     open(joinpath(dir, name), "a") do io
#                         write(io, ",")
#                     end
#                 end
#             end
#         end


#         for group in values(groups)
#             for name in group
#                 open(joinpath(dir, name), "a") do io
#                     write(io, "],")
#                 end
#             end
#         end
#     end
# end
# test_functions(4, 4, 8, 1) 

# dir = "zzz\\data"
# # filename = "file.txt"
# filename = joinpath(dir, "file.txt")
# open(filename, "a") do io
#     write(io, "New line of text\n")
# end



# struct Integrand{Op,LSt,LSb,Elt,Elb}
#     operator::Op
#     local_test_space::LSt
#     local_trial_space::LSb
#     test_chart::Elt
#     trial_chart::Elb
# end


# function (igd::Integrand)(u,v)
    
#     x = neighborhood(igd.test_chart,u)
#     y = neighborhood(igd.trial_chart,v)
    
#     f = igd.local_test_space(x)
#     g = igd.local_trial_space(y)

#     return jacobian(x) * jacobian(y) * igd(x,y,f,g)
# end
# function (igd::Integrand)(x,y,f,g)

#     op = igd.operator
#     kervals = kernelvals(op, x, y)
#     _integrands_leg(op, kervals, f, x, g, y)

# end
# function _integrands_leg_gen(f::Type{U}, g::Type{V}) where {U<:SVector{N}, V<:SVector{M}} where {M,N}
#     ex = :(SMatrix{N,M}(()))
#     for m in 1:M
#         for n in 1:N
#             push!(ex.args[2].args, :(integrand(op, kervals, f[$n], x, g[$m], y)))
#         end
#     end
#     return ex
# end
# @generated function _integrands_leg(op, kervals, f::SVector{N}, x, g::SVector{M}, y) where {M,N}
#     _integrands_leg_gen(f, g)
# end




# # Error Matrix
# @show M_ref
# @show M
# @show size(M)




    




# # # Statistics
# mae = mean(error_matrix)
# rmse = sqrt(mean(error_matrix .^ 2))
# max_abs_error = maximum(error_matrix)
# rel_error_matrix = abs.(M_ref .- M) ./ (abs.(M_ref) .+ 1e-12)  # Avoid divide-by-zero
# mean_rel_error = mean(rel_error_matrix)
# agreement_percentage = 100 * mean(error_matrix .<= 1e-8)

# # # Correlation
# # corr_coef = cor(vec(A), vec(B))

# 0.0134763-0.00232197im    0.0018405-0.0018905im    0.000854466-0.00182291im   0.00186389-0.00191576im   …    0.0005639-0.00115984im    0.00104559-0.00159947im  0.000527751-0.0014276im    0.000542131-0.0011498im
# 0.00184079-0.00189052im    0.0116707-0.00192067im    0.00178397-0.0018504im   0.000658317-0.00154041im      0.000310965-0.000987858im  0.000709358-0.00139404im  0.000673945-0.00135987im   0.000791178-0.00111629im
# 0.000854444-0.00182291im    0.0017843-0.0018504im      0.0130768-0.00222963im    0.0018203-0.00187629im      0.000541134-0.00113409im   0.000543093-0.00143372im   0.00101809-0.00153716im   0.000553164-0.00113159im
# 0.00186354-0.00191576im  0.000658339-0.00154041im    0.00182059-0.00187631im    0.0119126-0.00197205im      0.000800754-0.00113585im   0.000695822-0.00140742im  0.000693149-0.00137989im   0.000309456-0.000994595im
# 0.00195198-0.00205572im  0.000788151-0.00167767im   0.000267555-0.00159554im  0.000720803-0.00168174im       0.00180128-0.00132582im    0.00554891-0.00182784im   0.00165815-0.00164032im    0.00181939-0.00132132im
# 0.000672337-0.00159682im    0.0016206-0.00164695im   0.000745052-0.00158506im  0.000196803-0.00130415im   …  0.000901444-0.00109246im    0.00213673-0.00153148im   0.00213328-0.00150167im    0.00427764-0.00122565im
# 0.000255445-0.00154212im  0.000661009-0.00156965im    0.00181684-0.00191022im  0.000757475-0.00161153im       0.00176241-0.00126084im    0.00164249-0.00159263im   0.00525935-0.00169851im     0.0016909-0.00125009im
# 0.000796781-0.00169859im  0.000207389-0.00135305im   0.000704949-0.00164631im   0.00172906-0.00175267im       0.00445759-0.00129262im    0.00227069-0.00161005im   0.00216819-0.00157204im   0.000934581-0.0011402im
#           ⋮                                                                                              ⋱             ⋮
# 0.00557859-0.0018426im   0.000961517-0.00145609im   0.000752631-0.00150052im   0.00226111-0.00162237im       0.00100494-0.00104201im    0.00147198-0.00138909im  0.000832102-0.00126337im   0.000653963-0.000983119im
# 0.00167042-0.0016538im     0.0004746-0.00129591im    0.00072041-0.00146488im   0.00224959-0.0015903im        0.00220327-0.0011013im     0.00208898-0.00141045im   0.00139896-0.00132984im   0.000808074-0.000992851im
# 0.00184316-0.00133286im  0.000492825-0.00104516im   0.000832856-0.00119994im    0.0044639-0.00129796im      0.000807631-0.000798382im  0.000829506-0.00101487im  0.000636131-0.000961405im    0.0003591-0.000711929im
# 0.000563593-0.00115999im  0.000310611-0.000987967im   0.00054089-0.00113424im  0.000800714-0.001136im     …   0.00605812-0.000887631im   0.00343773-0.00113676im   0.00336351-0.00111228im    0.00128765-0.000830528im
# 0.00104525-0.00159962im  0.000708727-0.00139414im   0.000542562-0.00143383im  0.000695269-0.00140753im       0.00343768-0.00113674im    0.00906779-0.00152553im   0.00378556-0.00144426im    0.00344468-0.00113119im
# 0.000527214-0.00142771im  0.000673431-0.00135998im    0.00101777-0.00153731im  0.000692556-0.00138im          0.00336333-0.00111226im    0.00378506-0.00144426im   0.00879373-0.00146216im    0.00336759-0.00110689im
# 0.000541893-0.00114996im  0.000791128-0.00111644im   0.000552851-0.00113174im  0.000309101-0.000994705im      0.00128756-0.000830528im   0.00344464-0.00113122im   0.00336755-0.00110691im    0.00601407-0.000878991im