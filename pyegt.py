import math
import numpy as np
from operator import itemgetter
from pyegt_cython import _fixation_probability_matrix
from pyegt_cython import _fixation_probability_pfunction


class MoranProcess:
    """
    Moran Process
    """

    def __init__(self, intensity_of_selection, population_size, mutation_probability, payoff_function=None,
                 game_matrix=None, number_of_strategies=None, mapping='EXP', **kwargs):
        """
        Creates a Moran process with given parameters
        :param intensity_of_selection:
        :param population_size:
        :param payoff_function:
        :param game_matrix:
        :param number_of_strategies:
        :param mapping:
        :param kwargs:
        """
        self.kwargs = kwargs
        if payoff_function is not None and game_matrix is None:
            if number_of_strategies is None:
                raise ValueError('When using a custom payoff_function you must specify number_of_strategies.')
            self.using_matrix = False
            self.number_of_strategies = number_of_strategies
        elif game_matrix is not None and payoff_function is None:
            # check types and shapes
            assert type(game_matrix) == np.ndarray, "game_matrix should be a square numpy array"
            assert game_matrix.shape[0] == game_matrix.shape[1], "game_matrix should be a square numpy array"
            assert len(game_matrix.shape) == 2, "game_matrix should be a square numpy array"
            # initialise variables
            self.number_of_strategies = game_matrix.shape[0]
            self.using_matrix = True

        else:
            raise ValueError('No valid payoff structure given, please specify a game_matrix or a payoff_function.')

        self.intensity_of_selection = intensity_of_selection
        self.population_size = population_size
        self.mapping = mapping
        self.game_matrix = game_matrix
        self.payoff_function = payoff_function
        self.mutation_probability = mutation_probability
        if self.mapping == "EXP":
            self.is_exponential_mapping = True
        else:
            self.is_exponential_mapping = False

    def fixation_probability(self, mutant_index, resident_index):
        if self.using_matrix:
            [[a, b], [c, d]] = self.game_matrix[[mutant_index, resident_index]][:, [mutant_index, resident_index]]

            return _fixation_probability_matrix(mutant_index, resident_index, a, b, c, d, self.population_size,
                                                self.is_exponential_mapping, self.intensity_of_selection)
        else:
            return _fixation_probability_pfunction(mutant_index, resident_index, self.payoff_function,
                                                   self.number_of_strategies,
                                                   self.population_size, self.is_exponential_mapping,
                                                   self.intensity_of_selection, **self.kwargs)

    def monomorphous_transition_matrix(self):
        """
        Computes the associated markov chain (transition matrix), when mutations are assumed to be small.
        The approximation is accurate when there are no stable mixtures between any pair of strategies.

        Returns
        -------
        ans: ndarray, stochastic matrix

        """
        ans = np.zeros((self.number_of_strategies, self.number_of_strategies))
        for i in range(0, self.number_of_strategies):
            for j in range(0, self.number_of_strategies):
                if i != j:
                    # chance that j appears in an i population
                    ans[i, j] = self.fixation_probability(j, i)
        ans *= self.mutation_probability / (self.number_of_strategies - 1)
        for i in range(0, self.number_of_strategies):
            ans[i, i] = 1.0 - math.fsum(ans[i, :])
        return ans


def stationary_distribution(transition_matrix):
    """
    Computes the stationary_distribution of a markov chain. The matrix is given by rows.

    Parameters
    ----------
    transition_matrix: ndarray (must be a numpy array)

    Returns
    -------
    out: ndarray

    Examples
    -------
    >>>stationary_distribution(np.array([[0.1,0.9],[0.9,0.1]]))
    Out[1]: array([ 0.5,  0.5])
    >>>stationary_distribution(np.array([[0.1,0.0],[0.9,0.1]]))
    Out[1]: array([ 1.,  0.])
    >>>stationary_distribution(np.array([[0.6,0.4],[0.2,0.8]]))
    Out[1]: array([ 0.33333333,  0.66666667])
    """
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    # builds a dictionary with position, eigenvalue
    # and retrieves from this, the index of the largest eigenvalue
    index = max(
        zip(range(0, len(eigenvalues)), eigenvalues), key=itemgetter(1))[0]
    # returns the normalized vector corresponding to the
    # index of the largest eigenvalue
    # and gets rid of potential complex values
    vector = np.real(eigenvectors[:, index])
    # normalise
    vector /= np.sum(vector, dtype=float)
    return vector
