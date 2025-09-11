import nltk
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def get_pt_stopwords():
    try:
        portuguese_stopwords = stopwords.words("portuguese")
    except:
        nltk.download("stopwords")
        portuguese_stopwords = stopwords.words("portuguese")
    return portuguese_stopwords


def get_text_processing_pipeline(portuguese_stopwords):
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=300,
                    min_df=7,
                    max_df=0.8,
                    stop_words=portuguese_stopwords,
                    ngram_range=(1, 2),
                ),
            )
        ]
    )
    return pipeline


def get_model(pipeline, df_path="processed_csvs/customer_reviews_preprocessed.csv"):
    df = pd.read_csv(df_path)

    if "comments" in df.columns:
        X = df["comments"].astype(str).tolist()

    y = df["sentiment"]

    X_prep = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y, test_size=0.20, random_state=42, stratify=y
    )

    logreg_param_grid = {
        "C": np.linspace(0.1, 10, 20),
        "penalty": ["l1", "l2"],
        "class_weight": ["balanced", None],
        "random_state": [42],
        "solver": ["liblinear"],
    }

    logreg = LogisticRegression()
    grid = GridSearchCV(
        logreg, logreg_param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    return grid, pipeline, (X_test, y_test)


def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig("metrics/confusion_matrix.png", dpi=300)
    plt.close()


def plot_classification_report(y_test, y_pred, labels):
    report = classification_report(
        y_test, y_pred, target_names=labels, output_dict=True
    )
    df = pd.DataFrame(report).transpose()

    df.iloc[:-3, :-1].plot(kind="bar", figsize=(8, 5))
    plt.title("Classification Report")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("metrics/classification_report.png", dpi=300)
    plt.close()


def predict_and_eval(grid, X_test, y_test):
    y_pred = grid.predict(X_test)
    print("Best Parameters:", grid.best_params_)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # get graphs
    labels = sorted(list(set(y_test)))
    plot_confusion_matrix(y_test, y_pred, labels)
    plot_classification_report(y_test, y_pred, labels)


def save_model(grid, pipeline):
    joblib.dump(grid.best_estimator_, "model/sentiment_model.pkl")
    joblib.dump(pipeline, "model/tfidf_pipeline.pkl")


if __name__ == "__main__":
    portuguese_stopwords = get_pt_stopwords()
    pipeline = get_text_processing_pipeline(portuguese_stopwords)
    model, pipeline, (X_test, y_test) = get_model(pipeline)
    predict_and_eval(model, X_test, y_test)
    save_model(model, pipeline)
