import streamlit as st
import time
import pandas as pd

from utils.backtracking import rezolva_tsp
from utils.nearest_neighbor import rezolva_tsp_nn, rezolva_tsp_nn_multistart
from utils.io_utils import load_dataset
from utils.hill_climbing_tsp import rezolva_hill_climbing
from utils.GA import rezolva_tsp_ga
from utils.simulated_annealing import rezolva_tsp_sa
from utils.performance import create_comparison_plot


# ==========================================
# ALGORITHM STRATEGIES (The Open/Closed Fix)
# ==========================================
# Each algorithm gets its own UI builder, parameter formatter, and executor.

def build_nn_ui():
    """Builds the sidebar user interface for the Nearest Neighbor algorithm.

        Returns:
            dict: A dictionary containing the configured parameters:
                - multistart (bool): Whether to use the multi-start variant.
                - start_node (int): The starting node index if multistart is False.
    """
    params = {}
    params["multistart"] = st.sidebar.checkbox("Use multistart", False)
    if not params["multistart"]:
        params["start_node"] = st.sidebar.number_input("Start node", 0, 100, 0)
    else:
        params["start_node"] = 0
    return params


def execute_nn(n, matrix, params):
    """Executes the Nearest Neighbor algorithm for the TSP.

        Args:
            n (int): The number of cities.
            matrix (list of list of float): The distance matrix representing the graph.
            params (dict): Dictionary of parameters configured in build_nn_ui().

        Returns:
            dict: A dictionary containing:
                - route (list of int): The calculated path.
                - cost (float): The total distance of the route.
        """
    if params.get("multistart"):
        route, cost = rezolva_tsp_nn_multistart(n, matrix)
    else:
        route, cost = rezolva_tsp_nn(n, matrix, start=params.get("start_node", 0))
    return {"route": route, "cost": cost}


def build_bt_ui():
    """Builds the sidebar user interface for the Backtracking algorithm.

        Returns:
            dict: A dictionary containing the configured parameters:
                - mode (str): Stopping condition ('prima', 'toate', 'y_solutii', 'timp').
                - solutions (int): Target number of solutions (if mode is 'y_solutii').
                - max_time (float): Time limit in seconds (if mode is 'timp').
        """
    params = {}
    params["mode"] = st.sidebar.selectbox("Mode", ["prima", "toate", "y_solutii", "timp"])
    if params["mode"] == "y_solutii":
        params["solutions"] = st.sidebar.number_input("Solutions", 1, 100, 10)
    else:
        params["solutions"] = 0

    if params["mode"] == "timp":
        params["max_time"] = st.sidebar.number_input("Max time", 0.1, 10.0, 2.0)
    else:
        params["max_time"] = 2.0
    return params


def execute_bt(n, matrix, params):
    """Executes the Backtracking algorithm for the TSP.

        Args:
            n (int): The number of cities.
            matrix (list of list of float): The distance matrix representing the graph.
            params (dict): Dictionary of parameters configured in build_bt_ui().

        Returns:
            dict: A dictionary containing:
                - route (list of int): The calculated path.
                - cost (float): The total distance of the route.
        """
    bt_result = rezolva_tsp(
        n,
        matrix,
        mod=params["mode"],
        y_solutii=params["solutions"],
        timp_max=params["max_time"]
    )
    return {"route": bt_result["traseu"], "cost": bt_result["cost"]}


def build_hc_ui():
    """Builds the sidebar user interface for the Hill Climbing algorithm.

        Returns:
            dict: A dictionary containing the configured parameters:
                - restarts (int): Number of random restarts.
                - variant (str): Neighborhood generation method ('swap' or '2-opt').
                - init (str): Initial state generation ('random' or 'nearest_neighbor').
        """
    return {
        "restarts": st.sidebar.number_input("Random restarts", 1, 50, 1),
        "variant": st.sidebar.selectbox("Neighborhood", ["swap", "2-opt"]),
        "init": st.sidebar.selectbox("Initial state", ["random", "nearest_neighbor"])
    }


def execute_hc(n, matrix, params):
    """Executes the Hill Climbing algorithm for the TSP.

        Args:
            n (int): The number of cities.
            matrix (list of list of float): The distance matrix representing the graph.
            params (dict): Dictionary of parameters configured in build_hc_ui().

        Returns:
            dict: A dictionary containing:
                - route (list of int): The calculated path.
                - cost (float): The total distance of the route.
        """
    hc_result = rezolva_hill_climbing(
        matrix,
        n,
        restarts=params["restarts"],
        variant=params["variant"],
        init=params["init"]
    )
    return {"route": hc_result["traseu"], "cost": hc_result["cost"]}


def build_ga_ui():
    """Builds the sidebar user interface for the Genetic Algorithm.

        Returns:
            dict: A dictionary containing the configured parameters:
                - population_size (int): Size of the generation pool.
                - mutation_rate (float): Probability of mutation.
                - generations (int): Total number of generations to run.
                - selection_type (str): Selection method ('tournament', 'roulette', 'rank').
                - tournament_size (int): Size of tournament (if selection_type is 'tournament').
                - elitism (int): Number of top individuals kept automatically.
                - nn_seeding (bool): Whether to seed the initial population with NN results.
        """
    params = {}
    params["population_size"] = st.sidebar.number_input("Population size", 10, 500, 50, 10)
    params["mutation_rate"] = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.1, 0.01)
    params["generations"] = st.sidebar.number_input("Generations", 10, 2000, 300, 50)
    params["selection_type"] = st.sidebar.selectbox("Selection type", ["tournament", "roulette", "rank"])

    if params["selection_type"] == "tournament":
        params["tournament_size"] = st.sidebar.slider("Tournament size", 2, 10, 3)
    else:
        params["tournament_size"] = 3

    params["elitism"] = st.sidebar.slider("Elitism (kept best individuals)", 0, 10, 2)
    params["nn_seeding"] = st.sidebar.checkbox("Seed 20% with NN solutions", value=True)
    return params


def execute_ga(n, matrix, params):
    """Executes the Genetic Algorithm for the TSP.

        Args:
            n (int): The number of cities.
            matrix (list of list of float): The distance matrix representing the graph.
            params (dict): Dictionary of parameters configured in build_ga_ui().

        Returns:
            dict: A dictionary containing:
                - route (list of int): The calculated path.
                - cost (float): The total distance of the route.
        """
    ga_result = rezolva_tsp_ga(
        n,
        matrix,
        pop_size=params["population_size"],
        n_generatii=params["generations"],
        rata_mutatie=params["mutation_rate"],
        tip_selectie=params["selection_type"],
        k_tournament=params["tournament_size"],
        keep_elitism=params["elitism"],
        start_from_20nn=params["nn_seeding"]
    )
    return {"route": ga_result["traseu"], "cost": ga_result["cost"]}


def build_sa_ui():
    """Builds the sidebar user interface for Simulated Annealing.

        Returns:
            dict: A dictionary containing the configured parameters:
                - max_iter (int): Maximum number of iterations.
                - cooling_rate (float): The rate at which temperature decreases.
                - initial_temp (float): The starting temperature.
        """
    return {
        "max_iter": st.sidebar.number_input("Max iterations", 100, 100000, 1000),
        "cooling_rate": st.sidebar.number_input("Cooling rate",
            min_value=0.8,
            max_value=0.9999,
            value=0.995,
            step=0.0001,
            format="%.4f"),
        "initial_temp": st.sidebar.number_input("Initial temperature", 1.0, 10000.0, 100.0)
    }


def execute_sa(n, matrix, params):
    """Executes the Simulated Annealing algorithm for the TSP.

    Args:
        n (int): The number of cities.
        matrix (list of list of float): The distance matrix representing the graph.
        params (dict): Dictionary of parameters configured in build_sa_ui().

    Returns:
        dict: A dictionary containing:
            - route (list of int): The calculated path.
            - cost (float): The total distance of the route.
    """
    sa_result = rezolva_tsp_sa(
        n=n,
        matrice=matrix,
        initial_temp=params["initial_temp"],
        cooling_rate=params["cooling_rate"],
        max_iter=params["max_iter"]
    )

    return {"route": sa_result["traseu"], "cost": sa_result["cost"]}


# --- REGISTRY ---
# To add a new algorithm, simply add an entry here. The UI loop handles the rest.
ALGORITHM_REGISTRY = {
    "Nearest Neighbor": {
        "build_ui": build_nn_ui,
        "execute": execute_nn
    },
    "Backtracking": {
        "build_ui": build_bt_ui,
        "execute": execute_bt
    },
    "Hill Climbing": {
        "build_ui": build_hc_ui,
        "execute": execute_hc
    },
    "Genetic Algorithm": {
        "build_ui": build_ga_ui,
        "execute": execute_ga
    },
    "Simulated Annealing": {
        "build_ui": build_sa_ui,
        "execute": execute_sa
    }
}


# ==========================================
# CORE EXECUTION ENGINE
# ==========================================

def generate_unique_name(base_name, runs):
    """Generates a unique name for a new experiment run.

        Appends an incrementing integer to the base name if it already exists
        in the list of current runs to prevent naming collisions.

        Args:
            base_name (str): The desired base name (usually the algorithm name).
            runs (list of dict): The current list of active runs in session state.

        Returns:
            str: A unique run name.
        """
    existing_names = [r["name"] for r in runs]
    if base_name not in existing_names:
        return base_name

    i = 1
    while f"{base_name}{i}" in existing_names:
        i += 1
    return f"{base_name}{i}"


def run_single_algorithm(algo_name, n, matrix, params):
    """Routes execution to the correct algorithm and records execution time.

        Args:
            algo_name (str): The name of the algorithm matching a key in ALGORITHM_REGISTRY.
            n (int): The number of cities.
            matrix (list of list of float): The distance matrix.
            params (dict): The configuration parameters for the algorithm.

        Returns:
            dict: A dictionary containing:
                - route (list of int): The calculated path.
                - cost (float): The total distance of the route.
                - time (float): The execution time in seconds.
        """
    start_time = time.perf_counter()

    # Dynamically fetch the correct execution function from the registry
    executor = ALGORITHM_REGISTRY[algo_name]["execute"]
    result = executor(n, matrix, params)

    result["time"] = time.perf_counter() - start_time
    return result


def execute_multiple_runs(algo_name, n, dataset_type, base_seed, nr_runs, params):
    """Executes a single algorithm multiple times and averages the results.

        Args:
            algo_name (str): The name of the algorithm to run.
            n (int): The number of cities.
            dataset_type (str): The type of dataset to generate ('Random Matrix' or 'Euclidean').
            base_seed (int): The starting seed for dataset generation.
            nr_runs (int): The number of times to execute the algorithm.
            params (dict): The configuration parameters for the algorithm.

        Returns:
            dict: A dictionary containing:
                - time (float): Average execution time across all runs.
                - cost (float or None): Average solution cost across all runs.
        """
    times = []
    costs = []

    for i in range(nr_runs):
        _, matrix = load_dataset(n, dataset_type, base_seed + i)

        result = run_single_algorithm(algo_name, n, matrix, params)
        times.append(result["time"])

        if result["cost"] is not None:
            costs.append(result["cost"])

    return {
        "time": sum(times) / len(times),
        "cost": sum(costs) / len(costs) if costs else None,
    }


# ==========================================
# UI RENDERING FUNCTIONS
# ==========================================

def render_scaling_results(dataset_type, seed, nr_runs, city_sizes):
    """Renders the scaling test experiment UI and plots.

        Executes all active runs over a range of city sizes to demonstrate
        how algorithmic runtime and cost scale with complexity. Plots are
        rendered using matplotlib/Streamlit pyplot.

        Args:
            dataset_type (str): The type of dataset being used.
            seed (int): The base seed for dataset generation.
            nr_runs (int): The number of repeated executions per data point.
            city_sizes (list of int): The array of city counts to test (the X-axis).
        """
    st.write("Running scaling test...")
    time_series = {}
    cost_series = {}

    for current_run in st.session_state.runs:
        algo = current_run["algo"]
        params_run = current_run["params"]
        name = current_run["name"]

        times, costs, valid_sizes = [], [], []

        for n in city_sizes:
            result = execute_multiple_runs(algo, n, dataset_type, seed, nr_runs, params_run)
            times.append(result["time"])
            costs.append(result["cost"])
            valid_sizes.append(n)

        time_series[name] = {"x": valid_sizes, "y": times}
        cost_series[name] = {"x": valid_sizes, "y": costs}

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


def render_comparison_results(n, dataset_type, seed, nr_runs):
    """Renders the fixed-size comparison experiment UI, tables, and charts.

        Executes all active runs for a single fixed city size and compares
        their performance and cost using dataframes, bar charts, and metrics.

        Args:
            n (int): The fixed number of cities.
            dataset_type (str): The type of dataset being used.
            seed (int): The base seed for dataset generation.
            nr_runs (int): The number of repeated executions for averaging.
        """
    st.write("Running comparison...")
    results = []

    for current_run in st.session_state.runs:
        result = execute_multiple_runs(
            algo_name=current_run["algo"],
            n=n,
            dataset_type=dataset_type,
            base_seed=seed,
            nr_runs=nr_runs,
            params=current_run["params"]
        )

        results.append({
            "Run Name": current_run["name"],
            "Algorithm": current_run["algo"],
            "Avg Cost": result["cost"],
            "Avg Time (s)": result["time"]
        })

    df = pd.DataFrame(results)
    st.subheader("Results")
    st.dataframe(df)

    time_df = df.sort_values("Avg Time (s)")
    st.bar_chart(time_df.set_index("Run Name")["Avg Time (s)"])

    cost_df = df.sort_values("Avg Cost")
    st.bar_chart(cost_df.set_index("Run Name")["Avg Cost"])

    best_cost = df.loc[df["Avg Cost"].idxmin()]
    best_time = df.loc[df["Avg Time (s)"].idxmin()]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best Solution", best_cost["Run Name"], f"Cost = {best_cost['Avg Cost']:.2f}")
    with col2:
        st.metric("Fastest Algorithm", best_time["Run Name"], f"{best_time['Avg Time (s)']:.4f}s")


# ==========================================
# MAIN APPLICATION LOOP
# ==========================================

def run():
    """Main entry point for the Streamlit application.

        Initializes session state, renders the global sidebar configurations,
        handles user additions/removals of algorithm runs, and triggers
        the appropriate experiment execution flows.
    """
    if "runs" not in st.session_state:
        st.session_state.runs = []

    st.title("TSP Algorithm Analyzer")
    st.write("Welcome to the TSP analysis application.")

    # --- Sidebar Setup ---
    mode = st.sidebar.radio(
        "Experiment mode",
        ["Algorithm comparison (fixed size)", "Scaling test (runtime vs cities)"]
    )

    st.sidebar.header("Dataset")
    dataset_type = st.sidebar.selectbox("Dataset type", ["Random Matrix", "Euclidean"])
    seed = st.sidebar.number_input("Dataset seed", min_value=0, value=1)
    nr_runs = st.sidebar.number_input("Number of runs per algorithm", min_value=1, value=1)

    if mode == "Algorithm comparison (fixed size)":
        num_cities = st.sidebar.slider("Number of cities", 5, 200, 30)
    else:
        min_cities = st.sidebar.slider("Min cities", 5, 50, 10)
        max_cities = st.sidebar.slider("Max cities", 10, 200, 50)
        step = st.sidebar.slider("Step", 1, 20, 5)

    # --- Experiment Builder ---
    st.sidebar.header("Experiment Builder")
    algo_type = st.sidebar.selectbox("Algorithm", list(ALGORITHM_REGISTRY.keys()))
    run_name = st.sidebar.text_input("Run name", value=algo_type)

    # Dynamically build UI based on the selected algorithm
    params = ALGORITHM_REGISTRY[algo_type]["build_ui"]()

    if st.sidebar.button("Add run"):
        unique_name = generate_unique_name(run_name, st.session_state.runs)
        st.session_state.runs.append({
            "name": unique_name,
            "algo": algo_type,
            "params": params.copy()
        })

    # --- Active Runs ---
    st.sidebar.divider()
    st.sidebar.header("Active Runs:")

    for i, current_run in enumerate(st.session_state.runs):
        with st.sidebar.container():
            st.markdown(f"### {current_run['name']}")
            st.caption(f"Algorithm: {current_run['algo']}")

            # Directly print English parameters
            for k, v in current_run["params"].items():
                # Formats key names visually (e.g. max_time -> Max time)
                formatted_key = str(k).replace("_", " ").capitalize()
                st.write(f"**{formatted_key}:** {v}")

            if st.button("Remove run", key=f"del_{i}"):
                st.session_state.runs.pop(i)
                st.rerun()
            st.divider()

    if st.sidebar.button("Clear all runs"):
        st.session_state.runs = []
        st.rerun()

    # --- Execution ---
    st.markdown("## Run experiments")
    start_comparison = st.button("Run comparison", type="primary")

    if start_comparison:
        if not st.session_state.runs:
            st.warning("Please add at least one run from the sidebar first.")
            return

        if mode == "Scaling test (runtime vs cities)":
            city_sizes = list(range(min_cities, max_cities + 1, step))
            render_scaling_results(dataset_type, seed, nr_runs, city_sizes)
        else:
            render_comparison_results(num_cities, dataset_type, seed, nr_runs)


if __name__ == "__main__":
    run()