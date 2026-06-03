import random
import time

from simpleai.search import SearchProblem, hill_climbing, hill_climbing_random_restarts
from utils.nearest_neighbor import rezolva_tsp_nn

class TSPHillClimbing(SearchProblem):
    """
    A Traveling Salesman Problem (TSP) formulation using Hill Climbing search.

    The problem represents cities as nodes and distances between them as a matrix.
    The goal is to find a permutation of cities that minimizes the total travel distance.
    """

    def __init__(self, orase, n, variant, init):
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

        self.variant = variant
        self.init = init

        self.actiuni = []
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                self.actiuni.append((i, j))

        self.initial_state = self.generate_random_state()

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

        if self.variant == "swap":
            for i in range(1, len(state) - 1):
                for j in range(i + 1, len(state)):
                    actions.append((i, j))

        elif self.variant == "2-opt":
            for i in range(1, len(state) - 2):
                for j in range(i + 1, len(state) - 1):
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
        i, j = action
        new_state = list(state)

        if self.variant == "swap":
            new_state[i], new_state[j] = new_state[j], new_state[i]

        elif self.variant == "2-opt":
            new_state[i:j + 1] = reversed(new_state[i:j + 1])

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

    def generate_initial_state(self):
        if self.init == "random":
            return self.generate_random_state()

        elif self.init == "nn":
            return self.generate_nn_state()

    def generate_nn_state(self):
        traseu, _ = rezolva_tsp_nn(self.n, self.orase, start=0)
        return traseu

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


def rezolva_hill_climbing(orase, nr_orase, restarts=1,
                          variant="swap",
                          init="random"):

    problem = TSPHillClimbing(
        orase,
        nr_orase,
        variant,
        init
    )

    start_time = time.perf_counter()

    global_best_state = None
    global_best_value = float("-inf")

    history = []

    for restart in range(restarts):

        current_state = tuple(problem.generate_initial_state())
        current_value = problem.value(current_state)

        if current_value > global_best_value:
            global_best_value = current_value
            global_best_state = current_state

            history.append(
                (
                    time.perf_counter() - start_time,
                    -global_best_value
                )
            )

        while True:

            best_neighbor = None
            best_neighbor_value = current_value

            for action in problem.getVecini(current_state):

                neighbor = problem.result(
                    current_state,
                    action
                )

                neighbor_value = problem.value(neighbor)

                if neighbor_value > best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

            # local optimum reached
            if best_neighbor is None:
                break

            current_state = best_neighbor
            current_value = best_neighbor_value

            # update global best
            if current_value > global_best_value:

                global_best_value = current_value
                global_best_state = current_state

                history.append(
                    (
                        time.perf_counter() - start_time,
                        -global_best_value
                    )
                )

    return {
        "traseu": list(global_best_state),
        "cost": -global_best_value,
        "history": history
    }


# def rezolva_hill_climbing(orase, nr_orase, restarts=1, variant="swap", init="random"):
#     """
#     Solves the TSP using hill climbing with random restarts.
#
#     Args:
#         orase (list[list[int]]): Distance matrix between cities.
#         nr_orase (int): Number of cities.
#
#     Returns:
#         tuple: A tuple containing:
#             - result: The final state found by the algorithm.
#             - result.value: The value (fitness) of the final state.
#     """
#     problem = TSPHillClimbing(orase, nr_orase, variant, init)
#
#     result = hill_climbing_random_restarts(
#         problem,
#         restarts_limit=restarts
#     )
#
#     return {
#         "traseu": result.state,
#         "cost": -result.value,   # because value is negative distance
#     }