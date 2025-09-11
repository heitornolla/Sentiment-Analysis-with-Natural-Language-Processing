import joblib
import pandas as pd
import lime
import lime.lime_text
import shap
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt


def load_model_and_pipeline():
    model = joblib.load("model/sentiment_model.pkl")
    vectorizer = joblib.load("model/tfidf_pipeline.pkl")
    return model, vectorizer


def run_lime_example(df, model, vectorizer, sample_idx=0):
    pipeline = make_pipeline(vectorizer, model)
    explainer = lime.lime_text.LimeTextExplainer(class_names=["negativo", "positive"])

    sample_text = df["comments"].iloc[sample_idx]

    exp = explainer.explain_instance(
        sample_text, pipeline.predict_proba, num_features=10
    )

    fig = exp.as_pyplot_figure()
    plt.savefig("explainability/lime.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_shap_example(df, model, vectorizer, sample_size=100):
    import matplotlib.pyplot as plt

    X_sample = df["comments"].sample(sample_size, random_state=42).tolist()
    X_transformed = vectorizer.transform(X_sample)

    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    # Run SHAP on the first review and get results
    shap.plots.bar(shap_values[0], max_display=10, show=False)
    plt.savefig("explainability/shap.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("processed_csvs/customer_reviews_preprocessed.csv")
    model, vectorizer = load_model_and_pipeline()

    # Run LIME on first review
    run_lime_example(df, model, vectorizer, sample_idx=0)

    # Run SHAP on sample of reviews
    run_shap_example(df, model, vectorizer, sample_size=100)
