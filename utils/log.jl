using Base.Threads: Atomic


function log_time(time_logger, key::String, value)
    if haskey(time_logger, key)
        push!(time_logger[key], value)
    else
        time_logger[key] = [value]
    end
end

mutable struct TimeLogger
    logger::Dict{String, Any}

    function TimeLogger()
        keys = [
            "time overhead",
            "time to determin the quadrule",
            "calculate the double int",
            "transfer quadrule to CPU",
            "calculate double for loop",
            "calculate SauterSchwab",
            "time_table[1,:]",
            "time_table[2,:]",
            "time_to_store",
            "transfer results to CPU",
            "create results as complex numbers",
            "time_sauter_schwab_overhead_and_test_toll 2",
            "time_sauter_schwab_overhead_and_test_toll 3",
            "time_sauter_schwab_overhead_and_test_toll 4",
            "calc_sauter_schwab 2",
            "calc_sauter_schwab 3",
            "calc_sauter_schwab 4"
        ]
        logger = Dict{String, Any}(key => [] for key in keys)
        new(logger)
    end
end

Base.getindex(tl::TimeLogger, key::String) = getindex(tl.logger, key)
Base.setindex!(tl::TimeLogger, value, key::String) = setindex!(tl.logger, value, key)
Base.haskey(tl::TimeLogger, key::String) = haskey(tl.logger, key)

function extract_atomic_values(value)
    if isempty(value)
        return []
    end

    # Atomic Float64
    if all(x -> isa(x, Atomic{Float64}), value)
        return mean([x[] for x in value])
    elseif all(x -> isa(x, AbstractVector{<:Atomic{Float64}}), value)
        return [[x[] for x in row] for row in value]
    elseif all(x -> isa(x, Float64), value)
        return mean(value)
    elseif all(x -> isa(x, AbstractVector{<:Float64}), value)
        return mean(vcat(value...))

    # Atomic Int64
    elseif all(x -> isa(x, Atomic{Int64}), value)
        return mean([x[] for x in value])
    elseif all(x -> isa(x, AbstractVector{<:Atomic{Int64}}), value)
        return [[x[] for x in row] for row in value]
    elseif all(x -> isa(x, Int64), value)
        return mean(value)
    elseif all(x -> isa(x, AbstractVector{<:Int64}), value)
        return mean(vcat(value...))

    else
        @show value
        @show typeof(value)
        error("Unsupported structure: expected Vector{Atomic{Float64}} or Vector{Vector{Atomic{Float64}}}")
    end
end

function print_means(logger::TimeLogger)
    for (key, value) in logger.logger
        means = extract_atomic_values(value)
        println(key, "     ", means)
    end
end



function log_to_config(config, timingInfo)
    if config.timeLogger !== nothing
        time_logger = config.timeLogger
        log_time(time_logger, "time overhead", timingInfo.time_overhead)
        log_time(time_logger, "time to determin the quadrule", timingInfo.time_quadrule_types)
        log_time(time_logger, "calculate the double int", timingInfo.time_double_int)
        log_time(time_logger, "transfer quadrule to CPU", timingInfo.time_transfer_to_CPU)
        log_time(time_logger, "calculate double for loop", timingInfo.time_double_forloop)
        log_time(time_logger, "calculate SauterSchwab", timingInfo.time_sauter_schwab)
        for i in 2:4
            log_time(time_logger, "time_sauter_schwab_overhead_and_test_toll $i", timingInfo.time_table[1,i])
            log_time(time_logger, "calc_sauter_schwab $i", timingInfo.time_table[2,i])
        end
        log_time(time_logger, "time_table[1,:]", timingInfo.time_table[1,:])
        log_time(time_logger, "time_table[2,:]", timingInfo.time_table[2,:])
        log_time(time_logger, "time_to_store", timingInfo.time_to_store)
    end
end