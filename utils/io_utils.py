from pathlib import Path

import numpy as np

def get_orase(nr_orase):
    """
    Generates a symmetric distance matrix for a given number of cities
    and saves it to a file.

    The distances are randomly generated integers between 1 and 99.
    The matrix is symmetric, and the diagonal remains zero.

    Args:
        nr_orase (int): Number of cities.

    Returns:
        str: Path to the generated file containing the distance matrix.
    """
    orase = np.zeros((nr_orase, nr_orase), dtype=int)
    np.random.seed(1)
    for i in range(nr_orase):
        for j in range(i+1, nr_orase):
            orase[i, j] = np.random.randint(1,100, dtype=int)
            orase[j, i] = orase[i, j]

    filename = str(nr_orase) + "_cities_file.txt"
    dirr = 'input/' + filename

    baza = Path(__file__).resolve().parent

    # Construieste calea completa
    fisier = baza / dirr

    with open(fisier, "w") as file:
        file.write(str(nr_orase) + "\n")
        for i in range(nr_orase):
            for j in range(nr_orase):
                file.write(str(orase[i, j]) + " ")
            file.write("\n")

    return dirr

def citeste_matrice(cale_fisier):
    """Citeste matricea de distante dintr-un fisier text.

    Formatul fisierului: prima linie contine N (numarul de orase),
    urmatoarele N linii contin cate N intregi separati prin spatii,
    reprezentand matricea de distante NxN.

    Args:
        cale_fisier: Calea catre fisierul de intrare (str).

    Returns:
        Un tuplu (n, matrice) unde n este numarul de orase (int) si matrice
        este o lista de liste de intregi de dimensiune NxN.

    Raises:
        FileNotFoundError: Daca fisierul nu exista la calea specificata.
        ValueError: Daca formatul fisierului este invalid.
    """

    baza = Path(__file__).resolve().parent

    # Construieste calea completa
    fisier = baza / cale_fisier

    with open(fisier, 'r') as f:
        linii = [linie.strip() for linie in f if linie.strip()]
    n = int(linii[0])
    matrice = [[int(x) for x in linii[i + 1].split()] for i in range(n)]
    return n, matrice


def calculeaza_matrice_distante(coord):
    """Calculează matricea de distanțe euclidiene între toate perechile de orașe."""
    n = len(coord)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = coord[i][0] - coord[j][0]
            dy = coord[i][1] - coord[j][1]
            dist[i][j] = np.sqrt(dx**2 + dy**2)
    return dist
