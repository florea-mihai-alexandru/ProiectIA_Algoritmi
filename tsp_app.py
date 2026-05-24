import streamlit as st
import time
import matplotlib.pyplot as plt
from utils.backtracking import rezolva_tsp
from utils.nearest_neighbor import *
from utils.io_utils import *
from utils.hill_climbing_tsp import *
from utils.GA import rezolva_tsp_ga
from utils.performance import create_comparison_plot

def run():
    st.title("TSP Algorithm Analyzer")

    st.write("Welcome to the TSP analysis application.")

    mode = st.sidebar.radio(
        "Experiment mode",
        [
            "Algorithm comparison (fixed size)",
            "Scaling test (runtime vs cities)"
        ]
    )

    if mode == "Algorithm comparison (fixed size)":
        num_cities = st.sidebar.slider("Number of cities", 5, 200, 30)

    if mode == "Scaling test (runtime vs cities)":
        min_cities = st.sidebar.slider("Min cities", 5, 50, 10)
        max_cities = st.sidebar.slider("Max cities", 10, 200, 50)
        step = st.sidebar.slider("Step", 1, 20, 5)

    stop_time = st.sidebar.number_input(
        "Stop time (seconds)",
        min_value=0.1,
        value=2.0,
        step=0.1
    )

    algorithms = st.sidebar.multiselect(
        "Algorithms",
        ["Nearest Neighbor", "Backtracking", "Simulated Annealing", "Genetic Algorithm", "Hill Climbing"],
        default=["Nearest Neighbor", "Simulated Annealing"]
    )

    params = {}

    if "Backtracking" in algorithms:
        with st.sidebar.expander("Backtracking"):

            backtracking_mode = st.selectbox(
                "Mode",
                ["prima", "toate", "y_solutii", "timp"]
            )

            params["backtracking"] = {
                "mod": backtracking_mode
            }

            if backtracking_mode == "y_solutii":
                params["backtracking"]["y_solutii"] = st.number_input(
                    "Number of solutions",
                    min_value=1,
                    value=10
                )

            if backtracking_mode == "timp":
                params["backtracking"]["timp_max"] = st.number_input(
                    "Maximum runtime (seconds)",
                    min_value=0.1,
                    value=5.0,
                    step=0.1
                )

    if "Nearest Neighbor" in algorithms:
        with st.sidebar.expander("Nearest Neighbor"):

            multistart = st.checkbox("Use multistart", value=False)

            params["nn"] = {
                "multistart": multistart
            }

            if not multistart and not mode == "Scaling test (runtime vs cities)":
                params["nn"]["start_node"] = st.number_input(
                    "Start node",
                    min_value=0,
                    max_value=max(0, num_cities - 1),
                    value=0
                )

    if "Simulated Annealing" in algorithms:
        with st.sidebar.expander("Simulated Annealing"):
            params["sa"] = {
                "max_iter": st.number_input("Max iterations", 100, 100000, 1000),
                "cooling_rate": st.slider("Cooling rate", 0.8, 0.999, 0.95),
                "initial_temp": st.number_input("Initial temperature", 1.0, 1000.0, 100.0)
            }

    if "Genetic Algorithm" in algorithms:
        with st.sidebar.expander("Genetic Algorithm"):
            params["ga"] = {
                "population_size": st.number_input("Population size", 10, 500, 50),
                "mutation_rate": st.slider("Mutation rate", 0.0, 1.0, 0.1)
            }

    if "Hill Climbing" in algorithms:
        with st.sidebar.expander("Hill Climbing"):
            params["hc"] = {
                "restarts": st.number_input(
                    "Random restarts",
                    min_value=1,
                    max_value=50,
                    value=1
                )
            }

    run = st.button("Run comparison")

    st.write(f"Selected algorithms: {algorithms}")

    if run:

        if mode == "Scaling test (runtime vs cities)":

            st.write("Running scaling test...")

            time_series = {}
            cost_series = {}

            city_sizes = list(range(min_cities, max_cities + 1, step))

            for algo in algorithms:

                times = []
                costs = []
                valid_sizes = []

                for n in city_sizes:

                    filename = get_orase(n)
                    m, matrice = citeste_matrice(filename)

                    start_time = time.perf_counter()

                    # -------------------------
                    # NEAREST NEIGHBOR
                    # -------------------------
                    if algo == "Nearest Neighbor":

                        if params["nn"]["multistart"]:
                            traseu, cost = rezolva_tsp_nn_multistart(n, matrice)
                        else:
                            traseu, cost = rezolva_tsp_nn(n, matrice)
                    # -------------------------
                    # BACKTRACKING
                    # -------------------------
                    elif algo == "Backtracking":

                        if n > 12:
                            continue

                        rezultat = rezolva_tsp(
                            n,
                            matrice,
                            mod=params["backtracking"]["mod"],
                            y_solutii=params["backtracking"].get("y_solutii", 0),
                            timp_max=params["backtracking"].get("timp_max", 2)
                        )

                        cost = rezultat["cost"] if rezultat["traseu"] else None
                    # -------------------------
                    # HILL CLIMBING
                    # -------------------------
                    elif algo == "Hill Climbing":

                        rezultat = rezolva_hill_climbing(
                            matrice,
                            n,
                            restarts=params["hc"]["restarts"]
                        )

                        cost = rezultat["cost"]
                    # -------------------------
                    # SIMULATED ANNEALING (placeholder for now)
                    # -------------------------
                    elif algo == "Simulated Annealing":
                        cost = None
                        time.sleep(0.01)
                    elif algo == "Genetic Algorithm":

                        rezultat = rezolva_tsp_ga(
                            n,
                            matrice,
                            pop_size=params["ga"]["population_size"],
                            rata_mutatie=int(params["ga"]["mutation_rate"] * 100),
                            n_generatii=300
                        )

                        cost = rezultat["cost"]

                    else:
                        continue

                    elapsed = time.perf_counter() - start_time

                    times.append(elapsed)
                    costs.append(cost)
                    valid_sizes.append(n)

                time_series[algo] = {
                    "x": valid_sizes,
                    "y": times
                }

                cost_series[algo] = {
                    "x": valid_sizes,
                    "y": costs
                }
            fig1 = create_comparison_plot(
                time_series,
                x_label="Number of cities",
                y_label="Execution time (s)",
                title="TSP Runtime Scaling"
            )

            st.pyplot(fig1)

            fig2 = create_comparison_plot(
                cost_series,
                x_label="Number of cities",
                y_label="Solution cost",
                title="TSP Solution Quality Comparison"
            )

            st.pyplot(fig2)
        else:
            if "Backtracking" in algorithms:

                st.write("Running Backtracking...")

                # Example:
                # n, matrice = citeste_matrice(num_cities)

                filename = get_orase(num_cities)
                n, matrice = citeste_matrice(filename)

                rezultat = rezolva_tsp(
                    n,
                    matrice,
                    mod=params["backtracking"]["mod"],
                    y_solutii=params["backtracking"].get("y_solutii", 0),
                    timp_max=params["backtracking"].get("timp_max", 5)
                )

                if rezultat["traseu"] is not None:

                    traseu_text = " -> ".join(map(str, rezultat["traseu"]))
                    traseu_text += f" -> {rezultat['traseu'][0]}"

                    st.success("Backtracking completed!")

                    st.write(f"### Best route")
                    st.write(traseu_text)

                    st.write(f"### Cost")
                    st.write(rezultat["cost"])

                    st.write(f"### Runtime")
                    st.write(f"{rezultat['durata']:.4f} seconds")

                    st.write(f"### Solutions found")
                    st.write(rezultat["solutii_gasite"])

                else:
                    st.warning("No solution found.")

            if "Nearest Neighbor" in algorithms:

                st.write("Running Nearest Neighbor...")

                filename = get_orase(num_cities)
                n, matrice = citeste_matrice(filename)

                start_time = time.perf_counter()

                if params["nn"]["multistart"]:

                    traseu, cost = rezolva_tsp_nn_multistart(
                        n,
                        matrice
                    )

                else:

                    traseu, cost = rezolva_tsp_nn(
                        n,
                        matrice,
                        start=params["nn"]["start_node"]
                    )

                durata = time.perf_counter() - start_time

                traseu_text = " -> ".join(map(str, traseu))

                st.success("Nearest Neighbor completed!")

                st.write("### Route")
                st.write(traseu_text)

                st.write("### Cost")
                st.write(cost)

                st.write("### Runtime")
                st.write(f"{durata:.6f} seconds")

            if "Hill Climbing" in algorithms:
                st.write("Running Hill Climbing...")

                filename = get_orase(num_cities)
                n, matrice = citeste_matrice(filename)

                start_time = time.perf_counter()

                rezultat = rezolva_hill_climbing(
                    matrice,
                    n,
                    restarts=params["hc"]["restarts"]
                )

                durata = time.perf_counter() - start_time

                traseu_text = " -> ".join(map(str, rezultat["traseu"]))
                traseu_text += f" -> {rezultat['traseu'][0]}"

                st.success("Hill Climbing completed!")

                st.write("### Route")
                st.write(traseu_text)

                st.write("### Cost")
                st.write(rezultat["cost"])

                st.write("### Runtime")
                st.write(f"{durata:.6f} seconds")