# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Initialize Parameters: Set initial weights (theta) to zero.

Compute Predictions: Calculate predictions using the sigmoid function on the weighted inputs.

Calculate Cost: Compute the cost using the cross-entropy loss function.

Update Weights: Adjust weights by subtracting the gradient of the cost with respect to each weight.

Repeat: Repeat steps 2–4 for a set number of iterations or until convergence is achieved.

## Program:
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:

![443114789-5f4f25af-dba7-49bd-91e1-a10bbe1ce290](https://github.com/user-attachments/assets/543d8407-5194-41be-81a3-56592b398316)

![443114869-fecc0157-4dbe-4760-a64f-3fd9b5690d99](https://github.com/user-attachments/assets/68e41641-d315-4dfc-a8c3-19de5c7451e9)

![443114955-783bf79b-d2c4-48df-bc7c-959f3e721da9](https://github.com/user-attachments/assets/8818d4e5-3e0b-4871-9c7c-8ba1736df2d8)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
