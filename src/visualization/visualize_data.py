
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_data(X, y):
    sns.pairplot(pd.concat([X, y.rename('species')], axis=1), hue='species')
    plt.show()
