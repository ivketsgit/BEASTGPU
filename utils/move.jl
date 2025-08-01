function move(backend, input)
    out = KernelAbstractions.allocate(backend, eltype(input), size(input))
    KernelAbstractions.copyto!(backend, out, input)
    return out
end

function move(backend, out, input)
    KernelAbstractions.copyto!(backend, out, input)
end
