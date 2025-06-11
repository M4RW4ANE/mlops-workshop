from sklearn.datasets import load_iris
import pandas as pd

data = load_iris(as_frame=True)
df = data.frame

df.to_csv('data/iris.csv', index=False)
