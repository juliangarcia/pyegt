import numpy as np
import pyegt

if __name__ == "__main__":
    game_matrix = np.random.randint(0, 10, 400).reshape((20, 20))
    population_size = 1000
    ios = 0.1
    mutation_probability = 0.001
    moran = pyegt.MoranProcess(intensity_of_selection=ios, population_size=population_size,
                               mutation_probability=mutation_probability, payoff_function=None, game_matrix=game_matrix)

    result = moran.monomorphous_transition_matrix()
    print(pyegt.stationary_distribution(result))
