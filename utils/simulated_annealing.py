import random
import math

def calculeaza_cost(traseu, matrice):
    """Calculates the total cost of a given route.

    Args:
        traseu (list of int): The sequence of cities.
        matrice (list of list of float): The distance matrix.

    Returns:
        float: The total cost of traversing the route.
    """
    cost = 0
    for i in range(len(traseu) - 1):
        cost += matrice[traseu[i]][traseu[i + 1]]
    return cost


def genereaza_vecin(traseu):
    """Generates a neighboring route by swapping two internal nodes.

    Args:
        traseu (list of int): The current sequence of cities.

    Returns:
        list of int: A new route with two cities swapped.
    """
    vecin = traseu[:]
    # We only swap inner nodes (indices 1 to length - 2) to keep start/end at 0
    i, j = random.sample(range(1, len(traseu) - 1), 2)
    vecin[i], vecin[j] = vecin[j], vecin[i]
    return vecin


def rezolva_tsp_sa(n, matrice, initial_temp=100.0, cooling_rate=0.95, max_iter=1000):
    """Executes the Simulated Annealing algorithm for the TSP.

    Args:
        n (int): The number of cities.
        matrice (list of list of float): The distance matrix.
        initial_temp (float): The starting temperature.
        cooling_rate (float): The multiplier to decrease temperature per iteration.
        max_iter (int): The maximum number of iterations.

    Returns:
        dict: A dictionary containing the best route ('traseu') and its cost ('cost').
    """
    # FIX: Properly shuffle intermediate nodes to avoid the slicing bug
    noduri = list(range(1, n))
    random.shuffle(noduri)
    traseu_curent = [0] + noduri + [0]

    cost_curent = calculeaza_cost(traseu_curent, matrice)

    traseu_best = traseu_curent[:]
    cost_best = cost_curent
    temperatura = initial_temp

    for _ in range(max_iter):
        vecin = genereaza_vecin(traseu_curent)
        cost_vecin = calculeaza_cost(vecin, matrice)
        diferenta = cost_vecin - cost_curent

        # Accept if better
        if diferenta < 0:
            traseu_curent = vecin
            cost_curent = cost_vecin
        else:
            # Accept if worse based on thermal probability
            probabilitate = math.exp(-diferenta / temperatura)
            if random.random() < probabilitate:
                traseu_curent = vecin
                cost_curent = cost_vecin

        # Track global best
        if cost_curent < cost_best:
            traseu_best = traseu_curent[:]
            cost_best = cost_curent

        # Cool down
        temperatura *= cooling_rate

        # Prevent unnecessary iterations if the system is "frozen"
        if temperatura < 0.001:
            break

    return {"traseu": traseu_best, "cost": cost_best}