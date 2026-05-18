def rezolva_tsp_nn(n, matrice, start=0):
    """
    Rezolva TSP folosind euristica Nearest Neighbor.

    Args:
        n (int): Numarul de orase.
        matrice (list): Matricea de distante.
        start (int): Orasul de start.

    Returns:
        tuple: (traseu, cost_total)
    """

    vizitat = [False] * n
    vizitat[start] = True

    traseu = [start]

    oras_curent = start
    cost_total = 0

    # Alegem urmatoarele n-1 orase
    for _ in range(n - 1):

        cel_mai_apropiat = -1
        distanta_minima = float("inf")

        # cautam cel mai apropiat oras nevizitat
        for oras in range(n):

            if not vizitat[oras]:

                distanta = matrice[oras_curent][oras]

                if distanta < distanta_minima:
                    distanta_minima = distanta
                    cel_mai_apropiat = oras

        traseu.append(cel_mai_apropiat)

        vizitat[cel_mai_apropiat] = True

        cost_total += distanta_minima

        oras_curent = cel_mai_apropiat

    # intoarcere la start
    cost_total += matrice[oras_curent][start]

    traseu.append(start)

    return traseu, cost_total


def rezolva_tsp_nn_multistart(n, matrice):
    traseu_optim, cost_total_optim = rezolva_tsp_nn(n, matrice, 2)
    for i in range(1,n):
        # print(i)
        traseu, cost_total = rezolva_tsp_nn(n, matrice, i)
        if cost_total < cost_total_optim:
            cost_total_optim = cost_total
            traseu_optim = traseu
    return traseu_optim, cost_total_optim