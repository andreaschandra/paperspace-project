"""Train model on paperspace workflow
"""
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def read_data(path):
    """[summary]

    Args:
        path (str): path to read csv data

    Returns:
        dataframe: data from csv path
    """
    data = pd.read_csv(path)
    return data


def create_model(dataframe):
    """building model for iris dataset

    Args:
        dataframe (dataframe): dataframe iris dataset with label

    Returns:
        float: accuracy score for model
    """
    x_data = dataframe.iloc[:, 0:3].values
    y_label = dataframe.iloc[:, 4].values
    nb_model = MultinomialNB()
    nb_model.fit(x_data, y_label)
    y_pred = nb_model.predict(x_data)
    accuracy = accuracy_score(y_label, y_pred)

    return nb_model, accuracy


d_data = read_data("data.csv"           )
model, score = create_model(d_data)

print("accuracy", score)

# update dump comment here
pickle.dump(model, open("/artifacts/model.pkl", "wb"))
