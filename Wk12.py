import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

iris = pd.read_csv("Iris.csv")

X = iris.drop(columns=["Id", "Species"])
y = iris["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

tree = DecisionTreeClassifier(max_depth=3, random_state=1)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred, average="macro"), 3))
print("Recall:", round(recall_score(y_test, pred, average="macro"), 3))
print(confusion_matrix(y_test, pred))
