
import pandas as pd
from sklearn.datasets import load_iris

def load_iris_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y
