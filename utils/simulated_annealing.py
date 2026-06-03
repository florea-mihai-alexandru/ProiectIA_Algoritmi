import random
import math
import time


def calculeaza_cost(traseu, matrice):
    cost = 0
    for i in range(len(traseu) - 1):
        cost += matrice[traseu[i]][traseu[i + 1]]
    return cost


def genereaza_vecin(traseu):
    vecin = traseu[:]
    i, j = random.sample(range(1, len(traseu) - 1), 2)
    vecin[i], vecin[j] = vecin[j], vecin[i]
    return vecin


def rezolva_tsp_sa(n, matrice,
                   initial_temp=100.0,
                   cooling_rate=0.95,
                   max_iter=1000):

    start_time = time.perf_counter()

    noduri = list(range(1, n))
    random.shuffle(noduri)

    traseu_curent = [0] + noduri + [0]
    cost_curent = calculeaza_cost(traseu_curent, matrice)

    traseu_best = traseu_curent[:]
    cost_best = cost_curent

    temperatura = initial_temp

    history = []

    # initial point
    history.append((0.0, cost_best))

    for i in range(max_iter):

        vecin = genereaza_vecin(traseu_curent)
        cost_vecin = calculeaza_cost(vecin, matrice)

        diferenta = cost_vecin - cost_curent

        # acceptance rule
        if diferenta < 0:
            traseu_curent = vecin
            cost_curent = cost_vecin
        else:
            if random.random() < math.exp(-diferenta / temperatura):
                traseu_curent = vecin
                cost_curent = cost_vecin

        # update best
        if cost_curent < cost_best:
            cost_best = cost_curent
            traseu_best = traseu_curent[:]

        if i % 10 == 0:
            history.append(
                (time.perf_counter() - start_time, cost_best)
            )

        temperatura *= cooling_rate

        if temperatura < 0.001:
            break

    return {
        "traseu": traseu_best,
        "cost": cost_best,
        "history": history
    }