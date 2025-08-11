import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("bank.csv", sep=';')

print("Columns in dataset:", data.columns)

X = data.drop("y", axis=1)
y = data["y"]

if y.dtype == 'object':
    y = y.map({'yes': 1, 'no': 0})

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(30, 15))
plot_tree(clf, filled=True, fontsize=5, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

