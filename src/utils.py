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

def print_statictis_of_hpo(hpo_list):
    print("Num of EHR has HPO %d/%d" % (np.sum([1 for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]), len(hpo_list)))
    print("Avg HPO for all %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Median HPO for all %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Avg HPO for those have %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    print("Median HPO for those have %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))




