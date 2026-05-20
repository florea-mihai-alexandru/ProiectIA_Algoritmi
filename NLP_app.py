import streamlit as st
import time
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def load_dataset():

    categories = [
        'sci.space',
        'rec.sport.baseball',
        'comp.graphics'
    ]

    dataset = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )

    return dataset.data, dataset.target


def run():

    st.title("NLP Text Classification Analyzer")

    st.write("Compare NLP vectorization and classification methods.")

    mode = st.sidebar.radio(
        "Experiment mode",
        [
            "Single experiment",
            "Method comparison"
        ]
    )

    vectorizers = st.sidebar.multiselect(
        "Vectorization methods",
        [
            "Bag of Words",
            "TF-IDF"
        ],
        default=["Bag of Words", "TF-IDF"]
    )

    classifiers = st.sidebar.multiselect(
        "Classifiers",
        [
            "Naive Bayes",
            "Logistic Regression"
        ],
        default=["Naive Bayes"]
    )

    test_size = st.sidebar.slider(
        "Test size",
        0.1,
        0.5,
        0.2
    )

    max_features = st.sidebar.slider(
        "Max features",
        100,
        10000,
        3000
    )

    run_button = st.button("Run NLP Analysis")

    st.write(f"Selected vectorizers: {vectorizers}")
    st.write(f"Selected classifiers: {classifiers}")

    if run_button:

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        texts, labels = load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=42
        )

        results = []

        for vec_name in vectorizers:

            if vec_name == "Bag of Words":
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    stop_words='english'
                )

            elif vec_name == "TF-IDF":
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english'
                )

            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            for clf_name in classifiers:

                start_time = time.time()

                if clf_name == "Naive Bayes":
                    model = MultinomialNB()

                elif clf_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)

                model.fit(X_train_vec, y_train)

                predictions = model.predict(X_test_vec)

                runtime = time.time() - start_time

                accuracy = accuracy_score(y_test, predictions)

                precision = precision_score(
                    y_test,
                    predictions,
                    average='weighted'
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average='weighted'
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average='weighted'
                )

                results.append({
                    "Method": f"{vec_name} + {clf_name}",
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Runtime": runtime
                })

        results_df = pd.DataFrame(results)

        st.success("NLP analysis completed!")

        tab1, tab2, tab3 = st.tabs(
            ["Summary", "Metrics", "Charts"]
        )

        with tab1:

            st.subheader("Experiment Results")

            st.dataframe(results_df)

        with tab2:

            for index, row in results_df.iterrows():

                st.write(f"### {row['Method']}")

                st.write(f"Accuracy: {row['Accuracy']:.4f}")
                st.write(f"Precision: {row['Precision']:.4f}")
                st.write(f"Recall: {row['Recall']:.4f}")
                st.write(f"F1 Score: {row['F1 Score']:.4f}")
                st.write(f"Runtime: {row['Runtime']:.4f} seconds")

        with tab3:

            fig, ax = plt.subplots()

            ax.bar(
                results_df["Method"],
                results_df["Accuracy"]
            )

            ax.set_title("Accuracy Comparison")
            ax.set_xlabel("Methods")
            ax.set_ylabel("Accuracy")

            plt.xticks(rotation=15)

            st.pyplot(fig)

            fig2, ax2 = plt.subplots()

            ax2.bar(
                results_df["Method"],
                results_df["F1 Score"]
            )

            ax2.set_title("F1 Score Comparison")
            ax2.set_xlabel("Methods")
            ax2.set_ylabel("F1 Score")

            plt.xticks(rotation=15)

            st.pyplot(fig2)
