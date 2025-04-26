abstract type doubleQuadRuleGpuStrategy end

abstract type doubleQuadRuleGpuStrategyOptimistic <: doubleQuadRuleGpuStrategy end
abstract type doubleQuadRuleGpuStrategyRepaire <: doubleQuadRuleGpuStrategy end
abstract type doubleQuadRuleGpuStrategyShouldCalculate <: doubleQuadRuleGpuStrategy end

struct doubleQuadRuleGpuStrategyOptimisticInstance <: doubleQuadRuleGpuStrategyOptimistic end
struct doubleQuadRuleGpuStrategyRepaireInstance <: doubleQuadRuleGpuStrategyRepaire end
struct doubleQuadRuleGpuStrategyShouldCalculateInstance <: doubleQuadRuleGpuStrategyShouldCalculate end

function create_should_calc(T::doubleQuadRuleGpuStrategyRepaire, size_qrule)
    return 0
end

function create_should_calc(T::doubleQuadRuleGpuStrategyOptimistic, size_qrule)
    return 0
end

function create_should_calc(T::doubleQuadRuleGpuStrategyShouldCalculate, size_qrule)
    return ones(UInt32, size_qrule, size_qrule)
    # fill(typemax(UInt32), ceil(Int, size_qrule / 32) * 32, size_qrule)  
end
