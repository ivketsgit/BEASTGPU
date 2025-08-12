function test(M, M_ref)
    min_M_row = Array{Float64}(undef, size(M)[1])
    @threads for col in 1:size(M)[1]
        min_M_row[col] = abs.(M_ref[col] .- M[col])
    end
    return maximum(min_M_row)# < 10^-10
end