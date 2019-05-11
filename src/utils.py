import numpy as np
import torch

def softmax(array):
    expsum = np.sum(np.exp(array))
    return np.exp(array) / expsum

def sigmoid(x):
  return 1/(1+np.exp(-x))

def load_model(model, model_file):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict, strict=True)

def load_model_rm_module(model, model_file):
    state_dict = torch.load(model_file)
    state_dict = {key.replace("module.", ""): state_dict[key] for key in state_dict}
    model.load_state_dict(state_dict, strict=True)

def load_model_add_module(model, model_file):
    state_dict = torch.load(model_file)
    state_dict = {"module." + key: state_dict[key] for key in state_dict}
    model.load_state_dict(state_dict, strict=True)



