import streamlit as st

st.title("Multi-Purpose Analysis Tool")

st.write("Choose a module:")

if st.button("TSP Analysis"):
    st.session_state.page = "tsp"

if st.button("NLP Analysis"):
    st.session_state.page = "nlp"

if "page" in st.session_state:

    if st.session_state.page == "tsp":
        import tsp_app
        tsp_app.run()

    elif st.session_state.page == "nlp":
        import NLP_app
        NLP_app.run()