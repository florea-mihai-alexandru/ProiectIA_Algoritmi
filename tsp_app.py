import streamlit as st
import time
from utils.backtracking import rezolva_tsp
from utils.nearest_neighbor import *
from utils.io_utils import *
from utils.hill_climbing_tsp import *
from utils.GA import rezolva_tsp_ga
from utils.performance import create_comparison_plot
import pandas as pd

def generate_unique_name(base_name, runs):
    existing_names = [r["name"] for r in runs]

    if base_name not in existing_names:
        return base_name

    i = 1
    while f"{base_name}{i}" in existing_names:
        i += 1

    return f"{base_name}{i}"

def format_run_params(algo, params):
    if algo == "Hill Climbing":
        return {
            "Variant": params.get("variant", "-"),
            "Init": params.get("init", "-"),
            "Restarts": params.get("restarts", "-")
        }

    elif algo == "Nearest Neighbor":
        return {
            "Multistart": params.get("multistart", False),
            "Start node": params.get("start_node", "-")
        }

    elif algo == "Backtracking":
        return {
            "Mode": params.get("mod", "-"),
            "Solutions": params.get("y_solutii", "-"),
            "Max time": params.get("timp_max", "-")
        }

    elif algo == "Genetic Algorithm":
        return {
            "Population": params.get("population_size", "-"),
            "Mutation rate": params.get("mutation_rate", "-")
        }

    elif algo == "Simulated Annealing":
        return {
            "Max iter": params.get("max_iter", "-"),
            "Cooling rate": params.get("cooling_rate", "-"),
            "Initial temp": params.get("initial_temp", "-")
        }

    return params  # fallback

def run():
    if "runs" not in st.session_state:
        st.session_state.runs = []

    st.title("TSP Algorithm Analyzer")

    st.write("Welcome to the TSP analysis application.")

    mode = st.sidebar.radio(
        "Experiment mode",
        [
            "Algorithm comparison (fixed size)",
            "Scaling test (runtime vs cities)"
        ]
    )

    st.sidebar.header("Dataset")

    dataset_type = st.sidebar.selectbox(
        "Dataset type",
        [
            "Random Matrix",
            "Euclidean"
        ]
    )

    seed = st.sidebar.number_input(
        "Dataset seed",
        min_value=0,
        value=1
    )

    if mode == "Algorithm comparison (fixed size)":
        num_cities = st.sidebar.slider("Number of cities", 5, 200, 30)

    if mode == "Scaling test (runtime vs cities)":
        min_cities = st.sidebar.slider("Min cities", 5, 50, 10)
        max_cities = st.sidebar.slider("Max cities", 10, 200, 50)
        step = st.sidebar.slider("Step", 1, 20, 5)

    st.sidebar.header("Experiment Builder")

    params = {}

    algo_type = st.sidebar.selectbox(
        "Algorithm",
        ["Nearest Neighbor", "Backtracking", "Hill Climbing", "Simulated Annealing", "Genetic Algorithm"]
    )

    run_name = st.sidebar.text_input(
        "Run name",
        value=algo_type
    )

    if algo_type == "Hill Climbing":
        params["restarts"] = st.sidebar.number_input("Random restarts", 1, 50, 1)

        params["variant"] = st.sidebar.selectbox("Neighborhood", ["swap", "2-opt"])

        params["init"] = st.sidebar.selectbox("Initial state", ["random", "nearest_neighbor"])
    elif algo_type == "Nearest Neighbor":

        params["multistart"] = st.sidebar.checkbox("Use multistart", False)

        if not params["multistart"]:
            params["start_node"] = st.sidebar.number_input("Start node", 0, 100, 0)
    elif algo_type == "Backtracking":

        params["mod"] = st.sidebar.selectbox("Mode", ["prima", "toate", "y_solutii", "timp"])

        if params["mod"] == "y_solutii":
            params["y_solutii"] = st.sidebar.number_input("Solutions", 1, 100, 10)

        if params["mod"] == "timp":
            params["timp_max"] = st.sidebar.number_input("Max time", 0.1, 10.0, 2.0)
    elif algo_type == "Genetic Algorithm":

        params["population_size"] = st.sidebar.number_input("Population size", 10, 500, 50)

        params["mutation_rate"] = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.1)
    elif algo_type == "Simulated Annealing":

        params["max_iter"] = st.sidebar.number_input("Max iterations", 100, 100000, 1000)

        params["cooling_rate"] = st.sidebar.slider("Cooling rate", 0.8, 0.999, 0.95)

        params["initial_temp"] = st.sidebar.number_input("Initial temperature", 1.0, 1000.0, 100.0)

    add = st.sidebar.button("Add run")

    if add:
        unique_name = generate_unique_name(run_name, st.session_state.runs)

        st.session_state.runs.append({
            "name": unique_name,
            "algo": algo_type,
            "params": params.copy()
        })

    st.sidebar.divider()
    st.sidebar.header("Active Runs:")

    for i, run in enumerate(st.session_state.runs):

        with st.sidebar.container():
            st.markdown(f"### {run['name']}")
            st.caption("Algorithm: " + run["algo"])

            pretty_params = format_run_params(run["algo"], run["params"])

            for k, v in pretty_params.items():
                st.write(f"**{k}:** {v}")

            if st.button(f"Remove run", key=f"del_{i}"):
                st.session_state.runs.pop(i)
                st.rerun()

            st.divider()

    if st.sidebar.button("Clear all runs"):
        st.session_state.runs = []
        st.rerun()

    st.markdown("## Run experiments")

    run = st.button("Run comparison", type="primary")

    if run:

        if mode == "Scaling test (runtime vs cities)":

            st.write("Running scaling test...")

            time_series = {}
            cost_series = {}

            city_sizes = list(range(min_cities, max_cities + 1, step))

            for run in st.session_state.runs:

                algo = run["algo"]
                params_run = run["params"]
                name = run["name"]

                times = []
                costs = []
                valid_sizes = []

                for n in city_sizes:

                    m, matrice = load_dataset(
                        n,
                        dataset_type,
                        seed
                    )

                    start_time = time.perf_counter()

                    if algo == "Nearest Neighbor":

                        if params_run.get("multistart", False):
                            traseu, cost = rezolva_tsp_nn_multistart(n, matrice)
                        else:
                            traseu, cost = rezolva_tsp_nn(n, matrice)

                    elif algo == "Backtracking":

                        rezultat = rezolva_tsp(
                            n,
                            matrice,
                            mod=params_run["mod"],
                            y_solutii=params_run.get("y_solutii", 0),
                            timp_max=params_run.get("timp_max", 2)
                        )
                        cost = rezultat["cost"] if rezultat["traseu"] else None

                    elif algo == "Hill Climbing":

                        rezultat = rezolva_hill_climbing(
                            matrice,
                            n,
                            restarts=params_run["restarts"],
                            variant=params_run["variant"],
                            init=params_run["init"]
                        )
                        cost = rezultat["cost"]

                    elif algo == "Genetic Algorithm":

                        rezultat = rezolva_tsp_ga(
                            n,
                            matrice,
                            pop_size=params_run["population_size"],
                            rata_mutatie=int(params_run["mutation_rate"] * 100),
                            n_generatii=300
                        )
                        cost = rezultat["cost"]

                    elif algo == "Simulated Annealing":
                        cost = None
                        time.sleep(0.01)

                    elapsed = time.perf_counter() - start_time

                    times.append(elapsed)
                    costs.append(cost)
                    valid_sizes.append(n)

                time_series[name] = {"x": valid_sizes, "y": times}
                cost_series[name] = {"x": valid_sizes, "y": costs}

            # summary = []
            #
            # for run in st.session_state.runs:
            #     summary.append({
            #         "Name": run["name"],
            #         "Algorithm": run["algo"],
            #         "Variant": run["params"].get("variant", "-"),
            #         "Init": run["params"].get("init", "-"),
            #         "Restarts": run["params"].get("restarts", "-")
            #     })
            #
            # st.dataframe(pd.DataFrame(summary))

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
            pass
            # if "Backtracking" in algorithms:
            #
            #     st.write("Running Backtracking...")
            #
            #     # Example:
            #     # n, matrice = citeste_matrice(num_cities)
            #
            #     m, matrice = load_dataset(
            #         num_cities,
            #         dataset_type,
            #         seed
            #     )
            #
            #     rezultat = rezolva_tsp(
            #         m,
            #         matrice,
            #         mod=params["backtracking"]["mod"],
            #         y_solutii=params["backtracking"].get("y_solutii", 0),
            #         timp_max=params["backtracking"].get("timp_max", 5)
            #     )
            #
            #     if rezultat["traseu"] is not None:
            #
            #         traseu_text = " -> ".join(map(str, rezultat["traseu"]))
            #         traseu_text += f" -> {rezultat['traseu'][0]}"
            #
            #         st.success("Backtracking completed!")
            #
            #         st.write(f"### Best route")
            #         st.write(traseu_text)
            #
            #         st.write(f"### Cost")
            #         st.write(rezultat["cost"])
            #
            #         st.write(f"### Runtime")
            #         st.write(f"{rezultat['durata']:.4f} seconds")
            #
            #         st.write(f"### Solutions found")
            #         st.write(rezultat["solutii_gasite"])
            #
            #     else:
            #         st.warning("No solution found.")
            #
            # if "Nearest Neighbor" in algorithms:
            #
            #     st.write("Running Nearest Neighbor...")
            #
            #     m, matrice = load_dataset(
            #         num_cities,
            #         dataset_type,
            #         seed
            #     )
            #
            #     start_time = time.perf_counter()
            #
            #     if params["nn"]["multistart"]:
            #
            #         traseu, cost = rezolva_tsp_nn_multistart(
            #             m,
            #             matrice
            #         )
            #
            #     else:
            #
            #         traseu, cost = rezolva_tsp_nn(
            #             m,
            #             matrice,
            #             start=params["nn"]["start_node"]
            #         )
            #
            #     durata = time.perf_counter() - start_time
            #
            #     traseu_text = " -> ".join(map(str, traseu))
            #
            #     st.success("Nearest Neighbor completed!")
            #
            #     st.write("### Route")
            #     st.write(traseu_text)
            #
            #     st.write("### Cost")
            #     st.write(cost)
            #
            #     st.write("### Runtime")
            #     st.write(f"{durata:.6f} seconds")
            #
            # if "Hill Climbing" in algorithms:
            #     st.write("Running Hill Climbing...")
            #
            #     m, matrice = load_dataset(
            #         num_cities,
            #         dataset_type,
            #         seed
            #     )
            #
            #     start_time = time.perf_counter()
            #
            #     rezultat = rezolva_hill_climbing(
            #         matrice,
            #         m,
            #         restarts=params["hc"]["restarts"],
            #         variant=params["hc"]["variant"],
            #         init=params["hc"]["init"]
            #     )
            #
            #     durata = time.perf_counter() - start_time
            #
            #     traseu_text = " -> ".join(map(str, rezultat["traseu"]))
            #     traseu_text += f" -> {rezultat['traseu'][0]}"
            #
            #     st.success("Hill Climbing completed!")
            #
            #     st.write("### Route")
            #     st.write(traseu_text)
            #
            #     st.write("### Cost")
            #     st.write(rezultat["cost"])
            #
            #     st.write("### Runtime")
            #     st.write(f"{durata:.6f} seconds")