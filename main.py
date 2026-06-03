import streamlit as st

st.title("Multi-Purpose Analysis Tool")

st.write("Choose a module:")

if st.button("TSP Analysis"):
    st.session_state.page = "tsp"

if st.button("NLP Analysis"):
    st.session_state.page = "nlp"

if st.button("Our Team"):
    st.session_state.page = "team"


if "page" in st.session_state:

    if st.session_state.page == "tsp":
        import tsp_app
        tsp_app.run()

    elif st.session_state.page == "nlp":
        import NLP_app
        NLP_app.run()

    elif st.session_state.page == "team":
        st.title("Our Team")

        st.subheader("Reject AI embrace Humanity")

        st.image("assets/RaieH.jpeg", width=420)
        col1, col2, col3 = st.columns(3)
        with col1:
            # st.image("assets/orange.jpg", width=200)
            st.subheader("Membru 1: Florea Mihai-Alexandru")

        with col2:
            # st.image("assets/prune.jpg", width=200)
            st.subheader("Membru 2: Cebotari Vlad")

        st.subheader("An 3, Disciplina Inteligență artificială")