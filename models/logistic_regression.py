import os
import pickle

with open("artifacts/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("artifacts/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)


def classify(text: str):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return {"label": pred}
