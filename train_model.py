import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib

X = np.load("features/features.npy")
y = np.load("features/labels.npy")

if len(X) == 0:
    raise SystemExit("No feature rows found. Run extract_features.py first.")

print(f"[Train] Stage started: samples={len(X)}")

is_labeled = y != -1
y_labeled = y[is_labeled]
X_labeled = X[is_labeled]

unique_labels = np.unique(y_labeled) if len(y_labeled) > 0 else np.array([])
print(f"[Train] Labeled samples={len(y_labeled)} | classes={unique_labels.tolist() if len(unique_labels) else []}")

if len(unique_labels) >= 2:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_labeled, y_labeled)
    payload = {"model_type": "classifier", "model": model}
    print("[Train] Model selected: RandomForestClassifier (supervised)")
elif len(X) >= 5:
    model = IsolationForest(
        n_estimators=300,
        contamination=0.15,
        random_state=42,
    )
    model.fit(X)
    payload = {"model_type": "isolation_forest", "model": model}
    print("[Train] Model selected: IsolationForest (unsupervised fallback)")
else:
    raise SystemExit(
        "Not enough data to train. Need at least 5 clips, or labeled positives and negatives."
    )

joblib.dump(payload, "highlight_model.pkl")
print("[Train] Stage finished: saved highlight_model.pkl")
