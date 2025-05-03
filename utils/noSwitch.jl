# # Global holder for arguments and closures
# const _noswitch_fn_ref = Ref{Function}(nothing)

# function _noswitch_runner()
#     f = _noswitch_fn_ref[]
#     @assert f isa Function "Expected a Function in _noswitch_fn_ref"
#     f()
# end

function _noswitch_runner()
    _noswitch_fn_ref[]()
end

# Use Any instead of Function to allow initializing with `nothing`
const _noswitch_fn_ref = Ref{Any}(nothing)


macro noswitch(block)
    quote
        _noswitch_fn_ref[] = () -> begin
            $(esc(block))
        end
        Core._call_in_world_total(
            ccall(:jl_get_world_counter, UInt64, ()),
            _noswitch_runner
        )
    end
end
