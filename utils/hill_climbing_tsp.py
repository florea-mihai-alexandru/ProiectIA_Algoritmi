import random

from simpleai.search import SearchProblem, hill_climbing, hill_climbing_random_restarts

class TSPHillClimbing(SearchProblem):
    """
    A Traveling Salesman Problem (TSP) formulation using Hill Climbing search.

    The problem represents cities as nodes and distances between them as a matrix.
    The goal is to find a permutation of cities that minimizes the total travel distance.
    """

    def __init__(self, orase, n):
        """
        Initializes the TSP problem.

        Args:
            orase (list[list[int]]): Distance matrix where orase[i][j] represents
                the distance from city i to city j.
            n (int): Number of cities.
        """
        super().__init__()
        self.orase = orase
        self.n = n

        self.actiuni = []
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                self.actiuni.append((i, j))

    def getVecini(self, state):
        """
        Generates all possible neighbor actions for a given state.

        Each neighbor is defined as a swap between two positions in the state.

        Args:
            state (list[int] or tuple[int]): Current permutation of cities.

        Returns:
            list[tuple[int, int]]: List of index pairs representing swap actions.
        """
        actions = []
        for i in range(1, len(state) - 1):
            for j in range(i + 1, len(state)):
                actions.append((i, j))

        return actions

    def actions(self, state):
        """
        Returns the list of valid actions for a given state.

        If the state is incomplete, returns all predefined swap actions.
        Otherwise, returns an empty list.

        Args:
            state (list[int] or tuple[int]): Current state.

        Returns:
            list[tuple[int, int]]: Available actions.
        """
        if len(state) < self.n + 1:
            return self.actiuni
            # return self.getVecini(state)
        else:
            return []

    def result(self, state, action):
        """
        Applies an action to a state and returns the resulting new state.

        The action consists of swapping two positions in the state.

        Args:
            state (list[int] or tuple[int]): Current state.
            action (tuple[int, int]): Indices to swap.

        Returns:
            tuple[int]: New state after applying the swap.
        """
        new_state = list(state)
        new_state[action[0]], new_state[action[1]] = new_state[action[1]], new_state[action[0]]

        return tuple(new_state)

    def value(self, state):
        """
        Evaluates a state by computing the negative total distance of the tour.

        The function returns the negative distance because hill climbing
        maximizes the objective function.

        Args:
            state (list[int] or tuple[int]): A permutation representing a tour.

        Returns:
            int: Negative total distance of the tour.
        """
        dist = 0
        for i in range(len(state) - 1):
            dist += self.orase[state[i]][state[i + 1]]

        dist += self.orase[state[-1]][state[0]]
        return -dist

    def generate_random_state(self):
        """
        Generates a random initial state for the TSP.

        The starting city is fixed as 0, and the remaining cities are shuffled.

        Returns:
            list[int]: Randomly generated state.
        """
        initial_state = [i for i in range(1, self.n)]
        random.shuffle(initial_state)
        return [0] + initial_state


def rezolva_hill_climbing(orase, nr_orase):
    """
    Solves the TSP using hill climbing with random restarts.

    Args:
        orase (list[list[int]]): Distance matrix between cities.
        nr_orase (int): Number of cities.

    Returns:
        tuple: A tuple containing:
            - result: The final state found by the algorithm.
            - result.value: The value (fitness) of the final state.
    """
    problem = TSPHillClimbing(n=nr_orase, orase=orase)
    result = hill_climbing_random_restarts(problem, restarts_limit=1)

    return result, result.value