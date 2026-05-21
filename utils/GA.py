# tsp_genetic.py - Rezolvarea TSP cu algoritmi genetici (PyGAD)
import pygad
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from io_utils import *

# ══════════════════════════════════════════════════════════════════
# 1. DEFINIREA PROBLEMEI - ORAȘE ȘI DISTANȚE
# ══════════════════════════════════════════════════════════════════

# Coordonate aproximative (x, y) în km față de un punct de referință
ORASE_TEST = {
    0: ("Cluj-Napoca",  (0,    0   )),
    1: ("Brasov",       (220, -130 )),
    2: ("Bucuresti",    (330, -175 )),
    3: ("Timisoara",    (-175, -75 )),
    4: ("Iasi",         (380,   55 )),
    5: ("Constanta",    (450, -225 )),
    6: ("Craiova",      (160, -230 )),
    7: ("Galati",       (430,  -55 )),
    8: ("Oradea",       (-95,   45 )),
    9: ("Sibiu",        (95,   -95 )),
}

N_ORASE = len(ORASE_TEST)
COORD = np.array([ORASE_TEST[i][1] for i in range(N_ORASE)], dtype=float)
NUME_ORASE = [ORASE_TEST[i][0] for i in range(N_ORASE)]


DIST_MATRIX = calculeaza_matrice_distante(COORD)
# DIST_MATRIX = []

def chage_dist(dist):
    global DIST_MATRIX
    DIST_MATRIX = dist


def distanta_ruta(solutie):
    """Calculează distanța totală a unei rute (ciclu complet, revenire la start)."""
    total = 0.0
    n = len(solutie)
    for i in range(n):
        total += DIST_MATRIX[int(solutie[i])][int(solutie[(i + 1) % n])]
    return total


# ══════════════════════════════════════════════════════════════════
# 2. FUNCȚIA DE FITNESS
# ══════════════════════════════════════════════════════════════════

def fitness_func(ga_instance, solutie, solutie_idx):
    """PyGAD maximizează fitness-ul → returnăm negativul distanței."""
    return -distanta_ruta(solutie)


# ══════════════════════════════════════════════════════════════════
# 3. OPERATORI PERSONALIZAȚI - CROSSOVER ȘI MUTAȚIE
# ══════════════════════════════════════════════════════════════════

def ox_crossover(parinti, offspring_size, ga_instance):
    """
    Order Crossover (OX): produce mereu permutări valide din doi părinți.

    Algoritmul:
    1. Copiază segmentul [cx1..cx2] din parent1 în copil.
    2. Completează pozițiile libere cu genele din parent2 (în ordinea lor),
       omițând genele deja prezente în copil.
    """
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        p1 = parinti[idx % parinti.shape[0]].astype(int).tolist()
        p2 = parinti[(idx + 1) % parinti.shape[0]].astype(int).tolist()
        n = len(p1)

        cx1, cx2 = sorted(random.sample(range(n), 2))

        copil = [-1] * n
        copil[cx1:cx2 + 1] = p1[cx1:cx2 + 1]

        set_segment = set(copil[cx1:cx2 + 1])
        gene_ramase = [g for g in p2 if g not in set_segment]
        pozitii_libere = [i for i in range(n) if copil[i] == -1]

        for pos, gena in zip(pozitii_libere, gene_ramase):
            copil[pos] = gena

        offspring.append(copil)
        idx += 1

    return np.array(offspring, dtype=int)


def swap_mutation(offspring, ga_instance):
    """
    Swap Mutation: cu probabilitate rata_mutatie, schimbă două gene aleatoare.
    Constrângerea de permutare este menținută prin construcție.
    """
    rata = ga_instance.mutation_percent_genes / 100.0
    for i in range(offspring.shape[0]):
        if random.random() < rata:
            n = offspring.shape[1]
            idx1, idx2 = random.sample(range(n), 2)
            temp = int(offspring[i][idx1])
            offspring[i][idx1] = offspring[i][idx2]
            offspring[i][idx2] = temp
    return offspring


# ══════════════════════════════════════════════════════════════════
# 4. GENERARE POPULAȚIE INIȚIALĂ
# ══════════════════════════════════════════════════════════════════

def genereaza_populatie(pop_size, n_orase):
    """Generează pop_size permutări aleatoare distincte ale celor n_orase orașe."""
    pop = []
    for _ in range(pop_size):
        perm = list(range(n_orase))
        random.shuffle(perm)
        pop.append(perm)
    return np.array(pop, dtype=int)


# ══════════════════════════════════════════════════════════════════
# 5. RULAREA ALGORITMULUI GENETIC
# ══════════════════════════════════════════════════════════════════

def ruleaza_ga(pop_size=100, n_generatii=500, rata_mutatie=50,
               tip_selectie="tournament", k_tournament=3,
               keep_elitism=2, verbose=True):
    """
    Configurează și rulează GA pentru TSP cu parametrii specificați.
    Returnează: (instanța GA, distanța celei mai bune soluții, durata în secunde)
    """

    filename = get_orase(200)

    n,matrix = citeste_matrice(filename)

    # chage_dist(matrix)
    global DIST_MATRIX
    print("matrix", matrix)
    print("dist", DIST_MATRIX)


    populatie_initiala = genereaza_populatie(pop_size, N_ORASE)

    ga_instance = pygad.GA(
        num_generations=n_generatii,
        num_parents_mating=max(2, pop_size // 2),
        fitness_func=fitness_func,
        initial_population=populatie_initiala,
        crossover_type=ox_crossover,
        mutation_type=swap_mutation,
        mutation_percent_genes=rata_mutatie,
        parent_selection_type=tip_selectie,
        K_tournament=k_tournament,
        keep_elitism=keep_elitism,
        keep_parents=0,
        suppress_warnings=True,
    )

    start = time.time()
    ga_instance.run()
    durata = time.time() - start

    solutie, fitness, _ = ga_instance.best_solution()
    distanta = -fitness

    if verbose:
        print(f"\n{'='*55}")
        print(f"Configurație: pop={pop_size}, gen={n_generatii}, mut={rata_mutatie}%")
        ruta_str = " → ".join(ORASE_TEST[int(c)][0] for c in solutie)
        print(f"Ruta: {ruta_str} → {ORASE_TEST[int(solutie[0])][0]}")
        print(f"Distanță totală: {distanta:.2f}")
        print(f"Timp execuție:   {durata:.2f}s")

    return ga_instance, distanta, durata


# ══════════════════════════════════════════════════════════════════
# 6. VIZUALIZARE
# ══════════════════════════════════════════════════════════════════

def plot_convergenta(ga_instance, titlu="Curba de convergență", ax=None):
    """
    Grafic: distanța celei mai bune soluții per generație (curba de convergență).
    Dacă ax este None, creează o figură nouă și o afișează.
    """
    distante = [-f for f in ga_instance.best_solutions_fitness]
    afiseaza_singur = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(distante, color='steelblue', linewidth=1.5, label='Cea mai bună soluție')
    ax.set_xlabel("Generație")
    ax.set_ylabel("Distanță totală")
    ax.set_title(titlu)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if afiseaza_singur:
        plt.tight_layout()
        plt.show()


def plot_ruta(solutie, titlu="Ruta găsită de AG"):
    """Grafic: harta orașelor cu ruta vizualizată prin săgeți."""
    solutie_int = [int(c) for c in solutie]
    ruta = solutie_int + [solutie_int[0]]

    fig, ax = plt.subplots(figsize=(12, 9))

    for i in range(len(ruta) - 1):
        x1, y1 = COORD[ruta[i]]
        x2, y2 = COORD[ruta[i + 1]]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=2)
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(i + 1), fontsize=7, color='gray', ha='center')

    ax.scatter(COORD[:, 0], COORD[:, 1], s=150, c='tomato', zorder=5)
    for i in range(N_ORASE):
        x, y = COORD[i]
        ax.annotate(ORASE_TEST[i][0], (x, y),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    distanta = distanta_ruta(solutie)
    ax.set_title(f"{titlu}\nDistanță totală: {distanta:.2f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════
# 7. STUDIU DE PARAMETRI - EXEMPLU COMPLET
# ══════════════════════════════════════════════════════════════════

def studiu_parametri_populatie():
    """
    Sarcina 2: Compară configurații cu mărimi de populație diferite.
    Generează curbe de convergență suprapuse și grafic comparativ.
    """
    valori_pop = [20, 50, 100, 200]
    culori = ['tomato', 'steelblue', 'seagreen', 'darkorange']

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    distante_finale = []
    durate = []

    for pop, culoare in zip(valori_pop, culori):
        ga, dist, durata = ruleaza_ga(
            pop_size=pop, n_generatii=300, rata_mutatie=40, verbose=False
        )
        distante_finale.append(dist)
        durate.append(durata)

        d = [-f for f in ga.best_solutions_fitness]
        axes[0].plot(d, color=culoare, linewidth=1.5, label=f"pop={pop}")

    axes[0].set_xlabel("Generație")
    axes[0].set_ylabel("Distanță totală")
    axes[0].set_title("Curbe de convergență - mărimi de populație diferite")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar([str(p) for p in valori_pop], distante_finale,
                color=culori, alpha=0.85)
    axes[1].set_xlabel("Mărimea populației (sol_per_pop)")
    axes[1].set_ylabel("Distanță finală (mai mică = mai bun)")
    axes[1].set_title("Distanță finală în funcție de mărimea populației")
    for j, (v, d) in enumerate(zip(valori_pop, distante_finale)):
        axes[1].text(j, d + 2, f"{d:.1f}", ha='center', va='bottom', fontsize=9)

    axes[2].bar([str(p) for p in valori_pop], durate, color=culori, alpha=0.85)
    axes[2].set_xlabel("Mărimea populației (sol_per_pop)")
    axes[2].set_ylabel("Timp de execuție (s)")
    axes[2].set_title("Timp de execuție în funcție de mărimea populației")
    for j, (v, t) in enumerate(zip(valori_pop, durate)):
        axes[2].text(j, t + 0.01, f"{t:.2f}s", ha='center', va='bottom', fontsize=9)

    plt.suptitle("Studiu: Impactul mărimii populației", fontsize=13)
    plt.tight_layout()
    plt.show()


def studiu_parametri_mutatie():
    """
    Sarcina 3: Compară configurații cu rate de mutație diferite.
    """
    valori_mut = [5, 20, 40, 60, 80, 95]
    culori = ['navy', 'royalblue', 'steelblue', 'seagreen', 'darkorange', 'tomato']

    fig, ax = plt.subplots(figsize=(12, 6))
    distante_finale = []

    for mut, culoare in zip(valori_mut, culori):
        ga, dist, _ = ruleaza_ga(
            pop_size=100, n_generatii=300, rata_mutatie=mut, verbose=False
        )
        distante_finale.append(dist)
        d = [-f for f in ga.best_solutions_fitness]
        ax.plot(d, color=culoare, linewidth=1.5, label=f"mut={mut}%")

    ax.set_xlabel("Generație")
    ax.set_ylabel("Distanță totală")
    ax.set_title("Curbe de convergență - rate de mutație diferite")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nRezultate studiu mutație:")
    for mut, dist in zip(valori_mut, distante_finale):
        print(f"  mut={mut:3d}%  →  distanță finală: {dist:.2f}")


def studiu_tip_selectie():
    """
    Sarcina 4: Compară strategii de selecție a părinților.
    """
    strategii = ["tournament", "rws", "rank", "sus"]
    culori = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    fig, ax = plt.subplots(figsize=(12, 6))
    rezultate = []

    for strategie, culoare in zip(strategii, culori):
        ga, dist, durata = ruleaza_ga(
            pop_size=100, n_generatii=300, rata_mutatie=40,
            tip_selectie=strategie, verbose=False
        )
        rezultate.append({"strategie": strategie, "distanta": dist, "durata": durata})
        d = [-f for f in ga.best_solutions_fitness]
        ax.plot(d, color=culoare, linewidth=1.5, label=strategie)

    ax.set_xlabel("Generație")
    ax.set_ylabel("Distanță totală")
    ax.set_title("Curbe de convergență - strategii de selecție diferite")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\n{:<15} {:>15} {:>12}".format("Strategie", "Distanță finală", "Timp (s)"))
    print("-" * 45)
    for r in rezultate:
        print(f"{r['strategie']:<15} {r['distanta']:>15.2f} {r['durata']:>12.2f}")


# ══════════════════════════════════════════════════════════════════
# 8. PUNCT DE INTRARE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=== Rulare de bază (pop=100, gen=500, mut=50%) ===")
    ga, dist, durata = ruleaza_ga(pop_size=100, n_generatii=500, rata_mutatie=50)
    solutie, _, _ = ga.best_solution()

    plot_convergenta(ga, titlu="Curba de convergență - configurație de bază")
    plot_ruta(solutie, titlu="Ruta găsită de algoritmul genetic")

    print("\n=== Studiu: impact mărime populație ===")
    studiu_parametri_populatie()

    print("\n=== Studiu: impact rată mutație ===")
    studiu_parametri_mutatie()

    print("\n=== Studiu: tip selecție părinți ===")
    studiu_tip_selectie()
