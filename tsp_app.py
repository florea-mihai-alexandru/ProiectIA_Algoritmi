import streamlit as st
import time
import matplotlib.pyplot as plt

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
        min_cities = st.sidebar.slider("Min cities", 5, 100, 10)
        max_cities = st.sidebar.slider("Max cities", 10, 500, 100)
        step = st.sidebar.slider("Step", 5, 50, 10)

    stop_time = st.sidebar.number_input(
        "Stop time (seconds)",
        min_value=0.1,
        value=2.0,
        step=0.1
    )

    algorithms = st.sidebar.multiselect(
        "Algorithms",
        ["Nearest Neighbor", "Backtracking", "Simulated Annealing", "Genetic Algorithm"],
        default=["Nearest Neighbor", "Simulated Annealing"]
    )

    params = {}

    if "Nearest Neighbor" in algorithms:
        with st.sidebar.expander("Nearest Neighbor"):
            params["nn"] = {
                "start_node": st.number_input("Start node", 0, 100, 0)
            }

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

    run = st.button("Run comparison")

    st.write(f"Selected algorithms: {algorithms}")

    if run:
        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        st.success("Algorithm completed!")

    # x = [10, 20, 30, 40]
    # y = [0.1, 0.5, 2.0, 7.5]
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(x, y)
    # ax.set_title("Runtime Comparison")
    # ax.set_xlabel("Number of Cities")
    # ax.set_ylabel("Time (seconds)")
    #
    # st.pyplot(fig)

    tab1, tab2, tab3 = st.tabs(["Summary", "Distance", "Routes"])

    with tab1:
        x = [10, 20, 30, 40]
        y = [0.1, 0.5, 2.0, 7.5]

        fig, ax = plt.subplots()

        ax.plot(x, y)
        ax.set_title("Runtime Comparison")
        ax.set_xlabel("Number of Cities")
        ax.set_ylabel("Time (seconds)")

        st.pyplot(fig)

    with tab2:
        pass

    with tab3:
        pass