import random
import math
import time


def calculeaza_cost(traseu, matrice):
    cost = 0

    for i in range(len(traseu) - 1):
        cost += matrice[traseu[i]][traseu[i + 1]]

    return cost


def nearest_neighbor_start(matrice):

    n = len(matrice)

    nevizitate = set(range(1, n))

    traseu = [0]
    curent = 0

    while nevizitate:

        urmator = min(
            nevizitate,
            key=lambda x: matrice[curent][x]
        )

        traseu.append(urmator)

        nevizitate.remove(urmator)

        curent = urmator

    traseu.append(0)

    return traseu


def genereaza_vecin_2opt(traseu):

    vecin = traseu[:]

    i, j = sorted(
        random.sample(
            range(1, len(traseu) - 1),
            2
        )
    )

    vecin[i:j + 1] = reversed(
        vecin[i:j + 1]
    )

    return vecin


def rezolva_tsp_sa(
        n,
        matrice,
        initial_temp=100.0,
        cooling_rate=0.995,
        max_iter=1000):

    start_time = time.perf_counter()

    traseu_curent = nearest_neighbor_start(
        matrice
    )

    cost_curent = calculeaza_cost(
        traseu_curent,
        matrice
    )

    traseu_best = traseu_curent[:]
    cost_best = cost_curent

    temperatura = initial_temp

    history = [
        (0.0, cost_best)
    ]

    accepted = 0

    for i in range(max_iter):

        vecin = genereaza_vecin_2opt(
            traseu_curent
        )

        cost_vecin = calculeaza_cost(
            vecin,
            matrice
        )

        delta = cost_vecin - cost_curent

        if delta < 0:

            traseu_curent = vecin
            cost_curent = cost_vecin

            accepted += 1

        else:

            try:
                prob = math.exp(
                    -delta / temperatura
                )
            except OverflowError:
                prob = 0

            if random.random() < prob:

                traseu_curent = vecin
                cost_curent = cost_vecin

                accepted += 1

        if cost_curent < cost_best:

            cost_best = cost_curent
            traseu_best = traseu_curent[:]

        if i % 10 == 0:

            history.append(
                (
                    time.perf_counter() - start_time,
                    cost_best
                )
            )

        temperatura *= cooling_rate

        if temperatura < 0.001:
            break

    acceptance_rate = (
        accepted / (i + 1)
        if i > 0 else 0
    )

    return {
        "traseu": traseu_best,
        "cost": cost_best,
        "history": history,
        "acceptance_rate": acceptance_rate
    }