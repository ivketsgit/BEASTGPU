const dtol = 2.220446049250313e-16 * 1.0e3
const xtol2_mul_16 = 0.2 * 0.2  * 16
const xtol2 = 0.2 * 0.2

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
    
    result[i,j] = (hits != 0) * hits + (hits == 0) * (dmin2 * max(k2_mul_16, 1/σ_volume[j]) < xtol2_mul_16 ) * 4
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

