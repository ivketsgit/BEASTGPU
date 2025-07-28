
include("graph_data.jl")


for inv_density_factor in density_values
    for name in ["fill", "quadrule", "momintegrals", "store"]
        open("data/CPUMultiThread/$(inv_density_factor)/" * name * ".txt",  "w") do file
            print("")
        end
    end
end
