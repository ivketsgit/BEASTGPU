include("../graph_data.jl")
GPU_full = "../GPU_full"
sort_list = ["sortCPU", "sortGPU"]
store_list = ["storeGPU", "storeCPU"]

println(pwd())

inv_density_factors = density_values
for sort in sort_list, store in store_list, inv_density_factor in inv_density_factors
    base_path = joinpath(GPU_full)
    
    # Create GPU_full if needed
    if !isdir(base_path)
        mkpath(base_path)
        println("Created: $base_path")
    end

    sort_path = joinpath(base_path, sort)
    if !isdir(sort_path)
        mkpath(sort_path)
        println("Created: $sort_path")
    end

    store_path = joinpath(sort_path, store)
    if !isdir(store_path)
        mkpath(store_path)
        println("Created: $store_path")
    end

    final_path = joinpath(store_path, string(inv_density_factor))
    if !isdir(final_path)
        mkpath(final_path)
        println("Created: $final_path")
    else
        println("Already exists: $final_path")
    end
end