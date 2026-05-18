import matplotlib.pyplot as plt
from hill_climbing_tsp import *
from nearest_neighbor import *
from backtracking import *
from io_utils import citeste_matrice, get_orase

import time

def create_comparison_plot(
    series,
    ax=None,
    x_label="X",
    y_label="Y",
    title="Comparison",
    log_scale=False,
    figsize=(8, 5),
    markers=True
):
    """
    Creates a comparison plot for multiple algorithms.

    Parameters
    ----------
    series : dict
        Format:
        {
            "Algorithm Name": {
                "x": [...],
                "y": [...]
            }
        }

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)


    for label, data in series.items():

        x_values = data["x"]
        y_values = data["y"]

        if len(x_values) != len(y_values):
            raise ValueError(
                f"Length mismatch for '{label}'"
            )

        marker = 'o' if markers else None

        if log_scale:
            ax.semilogy(x_values, y_values,
                        marker=marker,
                        label=label)
        else:
            ax.plot(x_values, y_values,
                    marker=marker,
                    label=label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid(True)
    ax.legend()

    if fig is not None:
        fig.tight_layout()

    return fig


def ruleaza_experiment():
    """
    Runs a performance comparison experiment between Backtracking
    and Hill Climbing approaches for solving the Traveling Salesman Problem (TSP).

    The function:
    - Generates random distance matrices for different numbers of cities.
    - Measures execution time for both algorithms.
    - Stores and prints the results.
    - Plots the performance comparison using both linear and logarithmic scales.
    - Saves the resulting plot as an image file.

    The results are displayed in two subplots:
        1. Linear scale comparison
        2. Logarithmic scale comparison

    Returns:
        None
    """
    valori_n_bt = [5, 7, 8, 10, 12, 14,15]
    valori_n_hc = [5, 7, 8, 10, 12, 14,15, 50, 75, 100, 125, 150, 175, 200]
    valori_n_nn = [5, 7, 8, 10, 12, 14,15, 50, 75, 100, 125, 150, 175, 200, 300, 400]

    timpi_bt = []
    timpi_hc = []
    timpi_nn = []

    costuri_bt = []
    costuri_hc = []
    costuri_nn = []

    # Backtracking
    for n in valori_n_bt:
        filename = get_orase(n)

        m, matrice = citeste_matrice(filename)

        start = time.perf_counter()
        traseu, cost = rezolva_tsp(m, matrice)
        durata = time.perf_counter() - start

        print(f"Backtracking cu N={n} -> {durata:.6f} sec")
        print(traseu, "cost = ", cost)
        timpi_bt.append(durata)
        costuri_bt.append(cost)

    for n in valori_n_hc:
        filename = get_orase(n)
        m, matrice = citeste_matrice(filename)

        start = time.perf_counter()
        traseu, cost = rezolva_hill_climbing(matrice, n)
        durata = time.perf_counter() - start

        print(f"Hill Climbing  N={n} -> {durata:.6f} sec")
        print(traseu, "cost = ", cost)

        timpi_hc.append(durata)
        costuri_hc.append(cost)

    for n in valori_n_nn:
        filename = get_orase(n)
        m, matrice = citeste_matrice(filename)
        start = time.perf_counter()
        traseu, cost = rezolva_tsp_nn(n, matrice)
        durata = time.perf_counter() - start

        print(f"Nearest neighbor multistart N={n} -> {durata:.6f} sec")
        print(traseu, "cost = ", cost)

        timpi_nn.append(durata)
        costuri_nn.append(cost)

    fig, axes = plt.subplots(2, 2, figsize=(12, 5))

    series = {
        "Backtracking": {
            "x": valori_n_bt,
            "y": timpi_bt
        },
        "Nearest Neighbor": {
            "x": valori_n_nn,
            "y": timpi_nn
        },
        "Hill Climbing": {
            "x": valori_n_hc,
            "y": timpi_hc
        }
    }

    fig_linear = create_comparison_plot(
        series=series,
        ax=axes[0][0],
        x_label="Numar orase (N)",
        y_label="Timp executie (sec)",
        title="Comparatie performanta",
        log_scale=False
    )

    fig_log = create_comparison_plot(
        series=series,
        ax=axes[0][1],
        x_label="Numar orase (N)",
        y_label="Timp executie (sec)",
        title="Comparatie performanta (log)",
        log_scale=True
    )

    plt.tight_layout()
    # plt.savefig("comparare_performanta.png")
    plt.show()


if __name__ == '__main__':
   ruleaza_experiment()

