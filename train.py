import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

def main():
    # 1) Load dataset
    # Use the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "student-mat.csv"), sep=";")
    # 2) Create target (Pass/Fail) from G3, then remove G3 from features
    df["Pass"] = (df["G3"] >= 10).astype(int)
    df = df.drop(columns=["G3"])

    # 3) Split X/y
    X = df.drop(columns=["Pass"])
    y = df["Pass"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Auto-detect categorical vs numeric columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 5) Preprocessing (safe for deployment)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # 6) Model
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    # 7) One pipeline = preprocessing + model (this is what we deploy)
    pipe = Pipeline([
        ("pre", pre),
        ("model", model),
    ])

    # 8) Train
    pipe.fit(X_train, y_train)

    # 9) Quick check (optional)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print("Test accuracy:", acc)

    # 10) Save pipeline + feature list
    joblib.dump({"pipeline": pipe, "features": X.columns.tolist()}, "pipeline.joblib")
    print("Saved pipeline.joblib ✅")

if __name__ == "__main__":
    main()
