from pathlib import Path
import numpy as np

def get_orase(nr_orase, seed):
    """
    Generates a symmetric distance matrix for a given number of cities
    and saves it to a file only if the file does not already exist.

    Args:
        nr_orase (int): Number of cities.
        seed (int): Random seed.

    Returns:
        str: Path to the dataset file.
    """

    filename = f"{nr_orase}_cities_file.txt"
    dirr = Path("input") / filename

    baza = Path(__file__).resolve().parent
    fisier = baza / dirr

    # If file already exists, return it directly

    # if fisier.exists():
    #     return str(dirr)

    # Generate matrix only if file doesn't exist
    orase = np.zeros((nr_orase, nr_orase), dtype=int)

    np.random.seed(seed)

    for i in range(nr_orase):
        for j in range(i + 1, nr_orase):
            orase[i, j] = np.random.randint(1, 1000)
            orase[j, i] = orase[i, j]

    # Create input directory if it doesn't exist
    fisier.parent.mkdir(parents=True, exist_ok=True)

    with open(fisier, "w") as file:
        file.write(f"{nr_orase}\n")

        for i in range(nr_orase):
            for j in range(nr_orase):
                file.write(f"{orase[i, j]} ")
            file.write("\n")

    return str(dirr)

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


def generate_random_matrix(n, seed=1):
    rng = np.random.default_rng(seed)

    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            d = rng.integers(1, 1000)

            matrix[i, j] = d
            matrix[j, i] = d

    return matrix.tolist()


def generate_euclidean_matrix(n, seed=1):
    rng = np.random.default_rng(seed)

    coords = rng.integers(
        0,
        1000,
        size=(n, 2)
    )

    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):

            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]

            dist = int(round(np.hypot(dx, dy)))

            matrix[i, j] = dist
            matrix[j, i] = dist

    return matrix.tolist()


def get_orase_euclidean(nr_orase, seed=555):
    """
    Generates a Euclidean TSP instance.

    Cities are random points in a 2D plane and distances are
    Euclidean distances between coordinates.

    The generated distance matrix is saved to a file only if
    it does not already exist.

    Args:
        nr_orase (int): Number of cities.
        seed (int): Random seed.

    Returns:
        str: Relative path to the dataset file.
    """

    filename = f"{nr_orase}_cities_euclidean.txt"
    dirr = Path("input") / filename

    baza = Path(__file__).resolve().parent
    fisier = baza / dirr

    if fisier.exists():
        return str(dirr)

    rng = np.random.default_rng(seed)

    # Generate random city coordinates
    coordinates = rng.integers(
        low=0,
        high=1000,
        size=(nr_orase, 2)
    )

    # Build distance matrix
    orase = np.zeros((nr_orase, nr_orase), dtype=int)

    for i in range(nr_orase):
        for j in range(i + 1, nr_orase):

            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]

            dist = int(round(
                np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            ))

            orase[i, j] = dist
            orase[j, i] = dist

    fisier.parent.mkdir(parents=True, exist_ok=True)

    with open(fisier, "w") as file:

        file.write(f"{nr_orase}\n")

        for i in range(nr_orase):
            for j in range(nr_orase):
                file.write(f"{orase[i][j]} ")
            file.write("\n")

    return str(dirr)


def load_dataset(n, dataset_type, seed=1):
    seed+=n

    if dataset_type == "Euclidean":
        return n, generate_euclidean_matrix(n, seed)

    return n, generate_random_matrix(n, seed)