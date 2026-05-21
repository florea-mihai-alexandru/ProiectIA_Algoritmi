"""Rezolvarea problemei comis-voiajorului prin backtracking recursiv.

Implementare de referinta monolitica: un singur fisier Python, fara module externe.
Algoritmul garanteaza gasirea traseului de cost minim (solutia optima globala).

Utilizare:
    python backtracking_ref.py <fisier_intrare>

Exemplu:
    python backtracking_ref.py orase.txt
"""

import sys
import time
from .io_utils import citeste_matrice, get_orase

# Variabile globale pentru solutia optima.
# Sunt resetate la inceputul fiecarei rulari in rezolva_tsp().

def _backtracking(matrice, n, oras_curent, vizitat, traseu, cost, cost_min, _traseu_optim, mod, sol_gasite, start_time, y_solutii, timp_max, stop):
    """Explorare recursiva a spatiului de solutii TSP prin backtracking.

    La fiecare apel recursiv se incearca extinderea traseului curent cu un
    oras nevizitat. Ramurile al caror cost partial depaseste minimul global
    cunoscut sunt abandonate imediat (prunere branch-and-bound).

    Args:
        matrice: Matricea de distante NxN (lista de liste de intregi).
        n: Numarul de orase (int).
        oras_curent: Indexul orasului in care ne aflam la pasul curent (int).
        vizitat: Lista de booleeni de lungime n; vizitat[i] este True daca
            orasul i a fost deja inclus in traseu.
        traseu: Lista cu orasele vizitate pana acum, in ordinea parcurgerii.
            Primul element este intotdeauna 0 (orasul de start).
        cost: Costul acumulat al traseului partial curent (int sau float).
    """

    #CONDITII STOP
    match mod:
        case "prima":
            if sol_gasite[0] == 1:
                stop[0] = True
                return
        case "toate":
            pass
        case "y_solutii":
            if sol_gasite[0] == y_solutii:
                stop[0] = True
                return
        case "timp":
            durata_bkt = time.perf_counter() - start_time
            if durata_bkt >= timp_max:
                print(durata_bkt, n)
                stop[0] = True
                return

    if stop[0]:
        return

    # Caz de baza: toate orasele au fost vizitate — inchidem turul.
    if len(traseu) == n:
        sol_gasite[0]+=1
        cost_total = cost + matrice[oras_curent][traseu[0]]
        if cost_total < cost_min[0]:
            cost_min[0] = cost_total
            _traseu_optim[:] = traseu[:]  # copie a listei curente
        return

    # Pas recursiv: incercam extinderea traseului cu fiecare oras nevizitat.
    for urmator in range(n):
        if vizitat[urmator]:
            continue

        cost_nou = cost + matrice[oras_curent][urmator]

        # Prunere: abandonam ramura daca costul partial nu poate imbunatati
        # solutia optima cunoscuta (toate distantele sunt strict pozitive).
        if cost_nou >= cost_min[0]:
            continue

        vizitat[urmator] = True
        traseu.append(urmator)

        _backtracking(matrice, n, urmator, vizitat, traseu, cost_nou, cost_min, _traseu_optim, mod, sol_gasite, start_time, y_solutii, timp_max, stop)

        if stop[0]:
            return

        # Revenire (backtrack): restauram starea pentru a explora alte ramuri.
        traseu.pop()
        vizitat[urmator] = False


def rezolva_tsp(
    n,
    matrice,
    mod="timp",
    y_solutii=0,
    timp_max=5
):
    """
    Solve TSP using recursive backtracking with branch-and-bound.
    """

    _cost_minim = [sys.maxsize]
    _traseu_optim = []

    vizitat = [False] * n
    vizitat[0] = True

    sol_gasite = [0]
    stop = [False]

    start = time.perf_counter()

    _backtracking(
        matrice,
        n,
        0,
        vizitat,
        [0],
        0,
        _cost_minim,
        _traseu_optim,
        mod,
        sol_gasite,
        start,
        y_solutii,
        timp_max,
        stop
    )

    durata = time.perf_counter() - start

    if _traseu_optim:
        return {
            "traseu": _traseu_optim,
            "cost": _cost_minim[0],
            "durata": durata,
            "solutii_gasite": sol_gasite[0]
        }

    return {
        "traseu": None,
        "cost": None,
        "durata": durata,
        "solutii_gasite": sol_gasite[0]
    }

if __name__ == "__main__":
    nr_orase = 109
    filepath = get_orase(nr_orase)
    n,mat = citeste_matrice(filepath)
    mod = "y_solutii"
    y_solutii = 100
    print (rezolva_tsp(n,mat,mod,y_solutii))


# def rezolva_tsp(n, matrice):
#     """Rezolva TSP prin backtracking recursiv cu prunere branch-and-bound.
#
#     Citeste datele din fisierul specificat, ruleaza algoritmul de backtracking
#     si afiseaza traseul optim, costul minim si timpul de executie.
#
#     Args:
#         cale_fisier: Calea catre fisierul text cu matricea de distante (str).
#     """
#
#     # n, matrice = citeste_matrice(cale_fisier)
#
#     # print(f"Numar de orase: {n}")
#     # print("Matricea de distante:")
#     # for rand in matrice:
#     #     print("  " + "  ".join(f"{val:4d}" for val in rand))
#     # print()
#
#     # Resetam variabilele globale pentru a permite apeluri repetate.
#     _cost_minim = [sys.maxsize]
#     _traseu_optim = []
#
#     # Fixam orasul de start la indexul 0 (optimizare pentru TSP simetric:
#     # elimina N rotatii echivalente ale aceluiasi tur).
#     vizitat = [False] * n
#     vizitat[0] = True
#
#     start = time.perf_counter()
#     _backtracking(matrice, n, 0, vizitat, [0], 0, _cost_minim, _traseu_optim, "timp", [0], start, 0, 5, [False])
#     durata = time.perf_counter() - start
#
#     if _traseu_optim:
#         sir_traseu = " -> ".join(map(str, _traseu_optim))
#         sir_traseu += f" -> {_traseu_optim[0]}"
#         return _traseu_optim, _cost_minim[0]
#     else:
#         return None, 0 VARIANTA BUNA

    # print(f"Timp de executie: {durata:.6f} secunde")
