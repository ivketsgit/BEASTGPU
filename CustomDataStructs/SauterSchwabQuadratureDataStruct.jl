using StaticArrays
abstract type CustomGpuData end
abstract type SauterSchwabQuadratureCommonVertexCustomGpuData <: CustomGpuData end
abstract type SauterSchwabQuadratureCommonEdgeCustomGpuData <: CustomGpuData end
abstract type SauterSchwabQuadratureCommonFaceCustomGpuData <: CustomGpuData end
struct SauterSchwabQuadratureCommonVertexCustomGpuDataInstance <: SauterSchwabQuadratureCommonVertexCustomGpuData end
struct SauterSchwabQuadratureCommonEdgeCustomGpuDataInstance <: SauterSchwabQuadratureCommonEdgeCustomGpuData end
struct SauterSchwabQuadratureCommonFaceCustomGpuDataInstance <: SauterSchwabQuadratureCommonFaceCustomGpuData end

@kwdef mutable struct SauterSchwabQuadrature_gpu_data{TypeMarker}    
    ichart1_vert::Vector{SVector{3, SVector{2, Float64}}} = []
    ichart2_vert::Vector{SVector{3, SVector{2, Float64}}} = []
    ichart1_tan::Vector{SVector{2, SVector{2, Float64}}} = []
    ichart2_tan::Vector{SVector{2, SVector{2, Float64}}} = []
    ichart1_vol::Vector{Float64} = []
    ichart2_vol::Vector{Float64} = []

    qps::Vector{SVector{4, SVector{2, Float64}}} = []
    store_index::Vector{SVector{2, Int64}} = []
end

function add_element(obj::SauterSchwabQuadrature_gpu_data, 
    ichart1_vert::SVector{3, SVector{2, Float64}}, ichart2_vert::SVector{3, SVector{2, Float64}}, 
    ichart1_tan::SVector{2, SVector{2, Float64}}, ichart2_tan::SVector{2, SVector{2, Float64}}, 
    ichart1_vol::Float64, ichart2_vol::Float64, 
    qps::SVector{4, SVector{2, Float64}}, store_index::SVector{2, Int64})
    push!(obj.ichart1_vert, ichart1_vert)
    push!(obj.ichart2_vert, ichart2_vert)
    push!(obj.ichart1_tan, ichart1_tan)
    push!(obj.ichart2_tan, ichart2_tan)
    push!(obj.ichart1_vol, ichart1_vol)
    push!(obj.ichart2_vol, ichart2_vol)
    
    push!(obj.qps, qps)
    push!(obj.store_index, store_index)
end


function add_element(obj::SauterSchwabQuadrature_gpu_data, 
    qps::SVector{4, SVector{2, Float64}}, store_index::SVector{2, Int64})
    push!(obj.qps, qps)
    push!(obj.store_index, store_index)
end

function reset!(data::SauterSchwabQuadrature_gpu_data)
    data.ichart1_vert = []
    data.ichart2_vert = []
    data.ichart1_tan = []
    data.ichart2_tan = []
    data.ichart1_vol = []
    data.ichart2_vol = []
    data.qps = []
    data.store_index = []
    return data
end
