import os
import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

d_data = pd.read_csv("data.csv")

x = d_data.iloc[:, 0:3].values

y = d_data.iloc[:, 4].values

nb = MultinomialNB()

nb.fit(x, y)

y_pred = nb.predict(x)

accuracy = accuracy_score(y, y_pred)

print("accuracy", accuracy)

## update dump comment here
pickle.dump(nb, open("/artifacts/model.pkl", "wb"))