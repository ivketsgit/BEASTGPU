function print_array(arr)
    @print("\n[")
    dims = size(arr)
    for i in 1:dims[1]
        for j in 1:dims[2]
            @print(arr[i, j], ", ")
        end
        @print("; ")
    end
    @print("]")
end