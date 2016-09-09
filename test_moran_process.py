import pyegt
import numpy as np


def prisoners_dilemma_equal_gains(benefit=2,cost=1):
    return prisoners_dilemma(reward=benefit-cost,sucker=-cost,temptation=benefit,punishment=0)


def prisoners_dilemma(reward=3.0,sucker=0.0,temptation=4.0,punishment=1.0):
    return np.array([[reward,sucker],[temptation,punishment]])


def allc_tft_alld(reward=3.0,sucker=0.0,temptation=4.0,punishment=1.0,continuation_probability=0.95):
    return np.array([[reward,reward,sucker],[reward,reward, sucker*(1.0-continuation_probability)+punishment*continuation_probability],[temptation,temptation*(1.0-continuation_probability)+punishment*continuation_probability, punishment]])


def test_fixation():
    intensity_of_selection = 0.1
    population_size = 50
    game_matrix = prisoners_dilemma_equal_gains()
    moran = pyegt.MoranProcess(intensity_of_selection=intensity_of_selection, population_size=population_size,
                               mutation_probability=0.0, payoff_function=None, game_matrix=game_matrix)

    fix = moran.fixation_probability(mutant_index=0, resident_index=1)
    np.testing.assert_almost_equal(fix, 0.0006059821672805591)


def test_fixation_neutral():
    intensity_of_selection = 0.0
    population_size = np.random.randint(50, 200)
    game_matrix = prisoners_dilemma_equal_gains()
    moran = pyegt.MoranProcess(intensity_of_selection=intensity_of_selection, population_size=population_size,
                               mutation_probability=0.0, payoff_function=None, game_matrix=game_matrix)

    fix = moran.fixation_probability(mutant_index=0, resident_index=1)
    np.testing.assert_almost_equal(fix, 1.0/population_size)


def test_fixation_neutral_function():

    def payoff_function(focal_index, population_composition):
        return 0.0

    intensity_of_selection = 0.0
    population_size = np.random.randint(50, 200)
    moran = pyegt.MoranProcess(intensity_of_selection=intensity_of_selection, population_size=population_size,
                               mutation_probability=0.0, payoff_function=payoff_function, number_of_strategies=2)
    fix = moran.fixation_probability(mutant_index=0, resident_index=1)
    np.testing.assert_almost_equal(fix, 1.0/population_size)


def test_matrix():
    game_matrix = allc_tft_alld()
    population_size = 50
    ios = 0.1
    mutation_probability = 0.001
    moran = pyegt.MoranProcess(intensity_of_selection=ios, population_size=population_size,
                               mutation_probability=mutation_probability, payoff_function=None, game_matrix=game_matrix)

    result = moran.monomorphous_transition_matrix()
    expected = np.array(
        [[9.99939406e-01, 1.00000000e-05, 5.05937414e-05], [1.00000000e-05, 9.99989700e-01, 3.00468293e-07],
         [2.79103633e-07, 2.21444512e-05, 9.99977576e-01]])
    np.testing.assert_allclose(result, expected)