import math
import numpy as np
import cython

# cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double _fixation_probability_matrix(int mutant_index, int resident_index, double a, double b, double c, double d,
                                 int population_size, bint is_exponential_mapping, double intensity_of_selection):
    #cdef np.ndarray
    suma = np.zeros(population_size, dtype=np.float64)
    cdef double gamma = 1.0
    try:
        for i in range(1, population_size):
            payoff_mutant = (a * (i - 1) + b * (population_size - i)) / (population_size - 1)
            payoff_resident = (c * i + d * (population_size - i - 1)) / (population_size - 1)
            if is_exponential_mapping:
                fitness_mutant = math.e ** (intensity_of_selection * payoff_mutant)
                fitness_resident = math.e ** (intensity_of_selection * payoff_resident)
            else:
                fitness_mutant = 1.0 - intensity_of_selection + intensity_of_selection * payoff_mutant
                fitness_resident = 1.0 - intensity_of_selection + intensity_of_selection * payoff_resident
            gamma *= (fitness_resident / fitness_mutant)
            suma[i] = gamma
        return 1 / math.fsum(suma)
    except OverflowError:
        return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def _fixation_probability_pfunction(int mutant_index, int resident_index, payoff_function, int number_of_strategies,
                                 int population_size, bint is_exponential_mapping, double intensity_of_selection, **kwargs):
    suma = np.zeros(population_size, dtype=np.float64)
    cdef double gamma = 1.0
    try:
        for i in range(1, population_size):
            strategies = np.zeros(number_of_strategies, dtype=int)
            strategies[mutant_index] = i
            strategies[resident_index] = population_size - i
            payoff_mutant = payoff_function(
            mutant_index, population_composition=strategies, **kwargs)
            payoff_resident = payoff_function(
            resident_index, population_composition=strategies, **kwargs)
            if is_exponential_mapping:
                fitness_mutant = math.e ** (intensity_of_selection * payoff_mutant)
                fitness_resident = math.e ** (intensity_of_selection * payoff_resident)
            else:
                fitness_mutant = 1.0 - intensity_of_selection + intensity_of_selection * payoff_mutant
                fitness_resident = 1.0 - intensity_of_selection + intensity_of_selection * payoff_resident
            gamma *= (fitness_resident / fitness_mutant)
            suma[i] = gamma
        return 1 / math.fsum(suma)
    except OverflowError:
        return 0.0