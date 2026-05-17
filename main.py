import streamlit as st
import time
import matplotlib.pyplot as plt

st.title("TSP Algorithm Analyzer")

st.write("Welcome to the TSP analysis application.")

algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    [
        "Nearest Neighbor",
        "Brute Force",
        "Genetic Algorithm"
    ]
)

num_cities = st.sidebar.slider(
    "Number of Cities",
    min_value=5,
    max_value=100,
    value=20
)

run_button = st.sidebar.button("Run Algorithm")

st.write(f"Selected algorithm: {algorithm}")
st.write(f"Number of cities: {num_cities}")

if run_button:
    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    st.success("Algorithm completed!")

x = [10, 20, 30, 40]
y = [0.1, 0.5, 2.0, 7.5]

fig, ax = plt.subplots()

ax.plot(x, y)
ax.set_title("Runtime Comparison")
ax.set_xlabel("Number of Cities")
ax.set_ylabel("Time (seconds)")

st.pyplot(fig)