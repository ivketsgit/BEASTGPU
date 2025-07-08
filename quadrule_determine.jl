const dtol = 2.220446049250313e-16 * 1.0e3
const xtol2_mul_16 = 0.2 * 0.2  * 16
const xtol2 = 0.2 * 0.2

function determine_quadrule_types(config, biop, elementAssemblyData, timingInfo)
    timingInfo.time_quadrule_types += @elapsed begin
        backend = config.backend
        type = config.floatType

        quadrule_types_gpu = KernelAbstractions.allocate(backend, Int8, elementAssemblyData.elements_length_tuple)
        
        abs2_mul_16 = abs2(biop.gamma) * 16
        floatmax_type = floatmax(type)
        
        trial_elements_vertices_matrix = elementAssemblyData.elements_data[4]
        trial_elements_volume_matrix = elementAssemblyData.elements_data[6]
        test_elements_vertices_matrix = elementAssemblyData.elements_data[1]

        sizes = KernelAbstractions.zeros(backend, Int64, 3)
        
        quadrule_determine_type(backend, 1024)(
            quadrule_types_gpu, sizes,
            abs2_mul_16,
            test_elements_vertices_matrix,
            trial_elements_vertices_matrix,
            trial_elements_volume_matrix,
            floatmax_type,
            ndrange = elementAssemblyData.elements_length_tuple
        )
        
        KernelAbstractions.synchronize(backend)
    end

    return quadrule_types_gpu, sizes
end

@kernel function permake_array_volume(out::AbstractArray, @Const(k2_mul_16::Float64), @Const(σ_volume::AbstractArray))
    j = @index(Global, Linear)
    out[j] = xtol2_mul_16/ max(k2_mul_16, 1/σ_volume[j])
end

@kernel function quadrule_determine_type(result::AbstractArray, @Const(k2_mul_16::Float64), @Const(τ::AbstractArray), @Const(σ::AbstractArray), @Const(σ_volume::AbstractArray), @Const(floatmax_type::Float64))
    i, j = @index(Global, NTuple)

    hits = 0
    dmin2 = floatmax_type 
    @inbounds for u1 in 1:3
        @inbounds for u2 in 1:3
            dx = τ[i, 1, u1] - σ[j, 1, u2]
            dy = τ[i, 2, u1] - σ[j, 2, u2]
            dz = τ[i, 3, u1] - σ[j, 3, u2]
            d2 = dx*dx + dy*dy + dz*dz

            # d2 = 0
            # @inbounds for u3 in 1:3
            #     d2 += (τ[i, u3, u1] - σ[j, u3, u2])^2
            # end

            dmin2 = min(dmin2, d2)
            hits += sqrt(d2) < dtol
        end
    end
    
    result[i,j] = hits + (hits == 0) * (dmin2 * max(k2_mul_16, 1/σ_volume[j]) < xtol2_mul_16 ) * 4
end


@kernel function quadrule_determine_type(result::AbstractArray, @Const(τ::AbstractArray), @Const(σ::AbstractArray), @Const(σ_volume::AbstractArray), @Const(floatmax_type::Float64))
    i, j = @index(Global, NTuple)

    hits = 0
    dmin2 = floatmax_type 
    @inbounds for u1 in 1:3
        @inbounds for u2 in 1:3
            dx = τ[i, 1, u1] - σ[j, 1, u2]
            dy = τ[i, 2, u1] - σ[j, 2, u2]
            dz = τ[i, 3, u1] - σ[j, 3, u2]
            d2 = dx*dx + dy*dy + dz*dz
            dmin2 = min(dmin2, d2)
            hits += sqrt(d2) < dtol
        end
    end
    
    result[i,j] = (hits == 0 ? (dmin2  < σ_volume[j]) * 4 : hits)
end



@kernel function quadrule_determine_type(result::AbstractArray, sizes::AbstractArray, @Const(k2_mul_16::Float64), @Const(τ::AbstractArray), @Const(σ::AbstractArray), @Const(σ_volume::AbstractArray), @Const(floatmax_type::Float64))
    i, j = @index(Global, NTuple)
    sizes_local = @localmem Int (8,3)
    local_index = @index(Local, Linear)
    if local_index <= 3
        for iter in 1:8
            sizes_local[iter, local_index] = 0
        end
    end
    @synchronize

    hits = 0
    dmin2 = floatmax_type 
    @inbounds for u1 in 1:3
        @inbounds for u2 in 1:3
            dx = τ[i, 1, u1] - σ[j, 1, u2]
            dy = τ[i, 2, u1] - σ[j, 2, u2]
            dz = τ[i, 3, u1] - σ[j, 3, u2]
            d2 = dx*dx + dy*dy + dz*dz
            dmin2 = min(dmin2, d2)
            hits += sqrt(d2) < dtol
        end
    end
    
    result[i,j] = hits + (hits == 0) * (dmin2 * max(k2_mul_16, 1/σ_volume[j]) < xtol2_mul_16 ) * 4

    if hits != 0
        @atomic sizes_local[local_index % 8 + 1, hits] += 1
    end

    @synchronize
    
    if local_index <= 3
        sum = 0
        for iter in 1:8
            sum += sizes_local[iter, local_index]
        end
        @atomic sizes[local_index] += sum
    end
end