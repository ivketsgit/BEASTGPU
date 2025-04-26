abstract type ShouldCalc end

abstract type ShouldCalcTrue <: ShouldCalc end
abstract type ShouldCalcFalse <: ShouldCalc end

struct ShouldCalcTrueInstance <: ShouldCalcTrue end
struct ShouldCalcFalseInstance <: ShouldCalcFalse end

function create_results_matrix_gpu(backend, type, elements_length_tuple, T::ShouldCalcTrueInstance)
    return KernelAbstractions.allocate(backend, type, elements_length_tuple)
end

function create_results_matrix_gpu(backend, type, elements_length_tuple, T::ShouldCalcFalseInstance)
    return 0
end