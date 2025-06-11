from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from joblib import dump
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess("data/titanic.csv")

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)


# Métriques
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"Accuracy : {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall   : {rec:.2f}")
print(f"F1-score : {f1:.2f}")

with open("models/score_titanic.txt", "w") as f:
    f.write(f"Accuracy : {acc:.2f}\n")
    f.write(f"Precision: {prec:.2f}\n")
    f.write(f"Recall   : {rec:.2f}\n")
    f.write(f"F1-score : {f1:.2f}\n")

dump(model, "models/model_titanic.joblib")