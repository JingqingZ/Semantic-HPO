import numpy as np

def softmax(array):
    expsum = np.sum(np.exp(array))
    return np.exp(array) / expsum

def sigmoid(x):
  return 1/(1+np.exp(-x))

