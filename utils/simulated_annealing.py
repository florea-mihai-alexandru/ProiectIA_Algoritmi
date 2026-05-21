import random
import math


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


def rezolva_tsp_sa(
        n,
        matrice,
        initial_temp=100,
        cooling_rate=0.95,
        max_iter=1000
):

    traseu_curent = list(range(n))
    traseu_curent.append(0)

    random.shuffle(traseu_curent[:-1])

    cost_curent = calculeaza_cost(
        traseu_curent,
        matrice
    )

    traseu_best = traseu_curent[:]
    cost_best = cost_curent

    temperatura = initial_temp

    for _ in range(max_iter):
        vecin = genereaza_vecin(traseu_curent)
        cost_vecin = calculeaza_cost(
            vecin,
            matrice
        )
        diferenta = cost_vecin - cost_curent

        if diferenta < 0:

            traseu_curent = vecin
            cost_curent = cost_vecin
        else:

            probabilitate = math.exp(
                -diferenta / temperatura
            )

            if random.random() < probabilitate:

                traseu_curent = vecin
                cost_curent = cost_vecin

        if cost_curent < cost_best:

            traseu_best = traseu_curent[:]
            cost_best = cost_curent

        temperatura *= cooling_rate

        if temperatura < 0.001:
            break

    return traseu_best, cost_best