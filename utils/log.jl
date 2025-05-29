function log_time(time_logger, key::String, value)
    if haskey(time_logger, key)
        push!(time_logger[key], value)
    else
        time_logger[key] = [value]
    end
end