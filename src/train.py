from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess("data/iris.csv")

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.2f}")

with open("models/score.txt", "w") as f:
    f.write(f"Accuracy: {acc:.2f}\n")

dump(model, "models/model.joblib")