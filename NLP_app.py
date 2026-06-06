# NLP_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

np.random.seed(42)
plt.style.use('default')


def run():
    st.title("🔤 NLP Classification Tool")
    st.write("Train on `train.csv` • Test on `test.csv`")

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("Upload **train.csv**", type=["csv"], key="train")
    with col2:
        test_file = st.file_uploader("Upload **test.csv**", type=["csv"], key="test")

    if train_file is None or test_file is None:
        st.info("Please upload both **train.csv** and **test.csv** files.")
        st.markdown("""
        **Expected format (both files):**
        - Column with text (e.g. `text`, `review`, `sentence`)
        - Column with labels (e.g. `label`, `sentiment`, `polarity`)
        """)
        return

    # Load datasets
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        st.success(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return

    # Column selection
    st.subheader("Column Selection")
    all_cols = train_df.columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("Text Column", options=all_cols, index=0)
    with col2:
        label_col = st.selectbox("Label Column", options=all_cols, index=1)

    if st.button("Start Training & Evaluation", type="primary"):
        with st.spinner("Training models..."):
            analyze_with_train_test(train_df, test_df, text_col, label_col)


def analyze_with_train_test(train_df, test_df, text_col, label_col):
    # Extract data
    X_train = train_df[text_col].astype(str)
    y_train = train_df[label_col]
    X_test = test_df[text_col].astype(str)
    y_test = test_df[label_col]

    # Label distribution
    st.subheader("Label Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    y_train.value_counts().plot(kind='bar', ax=ax[0], title="Train Labels")
    y_test.value_counts().plot(kind='bar', ax=ax[1], title="Test Labels")
    st.pyplot(fig)
    plt.close()

    # Vectorization
    st.info("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42, class_weight='balanced'
        )
    }

    results = {}
    confusion_matrices = {}

    progress_bar = st.progress(0)
    status = st.empty()

    for i, (name, model) in enumerate(models.items()):
        status.text(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': acc,
            'report': classification_report(y_test, y_pred, zero_division=0),
            'predictions': y_pred
        }
        confusion_matrices[name] = cm

        progress_bar.progress((i + 1) / len(models))

    # Results
    st.success("✅ Training & Evaluation Completed!")

    st.subheader("📊 Model Comparison")
    comparison_df = pd.DataFrame({
        name: {'Accuracy': res['accuracy']} for name, res in results.items()
    }).T
    st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)

    # Detailed reports
    st.subheader("📋 Detailed Reports")
    for name, res in results.items():
        with st.expander(f"{name} — Accuracy: {res['accuracy']:.4f}"):
            st.text(res['report'])

    # Confusion Matrices
    st.subheader("🌀 Confusion Matrices")
    plot_confusion_matrices(confusion_matrices, y_test, results)


def plot_confusion_matrices(confusion_matrices, y_test, results):
    class_names = sorted(y_test.unique())
    n = len(confusion_matrices)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.ravel() if n > 1 else [axes]

    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i])
        axes[i].set_title(f'{name}\nAccuracy: {results[name]["accuracy"]:.4f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    run()