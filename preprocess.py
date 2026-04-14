import numpy as np

def preprocess_data(X):
    X = X / 255.0
    return X