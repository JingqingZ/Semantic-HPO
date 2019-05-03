import numpy as np

def softmax(array):
    expsum = np.sum(np.exp(array))
    return np.exp(array) / expsum

