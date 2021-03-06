
train_epoch = 10
train_epoch_unsupervised = 10
train_batch_size = 32
test_batch_size = 64

save_step_4_pretrain = 5000
save_step_4_annotation = 5000
save_step_4_unsupervised = 5000
print_step = 100
print_step_unsupervised = 100

sequence_length = 64
vocabulary_size = 30000

learning_rate_4_pretrain = 3e-4
learning_rate_4_annotation = 1e-3
learning_rate_4_unsupervised = 3e-4
warmup_proportion = 0.1 # proportion of training to perform linear learning rate warmup for.

step_to_train_all_weights = 20000

# TODO: MIMIC data are not shuffled, not sure if shuffle is necessary, seems not
training_percentage = 0.7
testing_percentage = 0.3
assert training_percentage + testing_percentage == 1

import numpy as np
total_num_mimic_record = 52722
np.random.seed(41) # the seed is the average of four numbers (1-100) randomly selected by my friends
mimic_indices = np.arange(total_num_mimic_record)
np.random.shuffle(mimic_indices)

mimic_train_indices = mimic_indices[:int(total_num_mimic_record * training_percentage)]
mimic_test_indices = mimic_indices[-int(total_num_mimic_record * testing_percentage):]


outputs_dir = "../outputs/"
# TODO: change
# outputs_model_dir = outputs_dir + "models/"
outputs_model_dir = outputs_dir + "models_with_noise/"
# TODO: CNN
# outputs_model_dir = outputs_dir + "models_cnn_with_noise/"
outputs_results_dir = outputs_dir + "results/"
outputs_figures_dir = outputs_dir + "figures/"
outputs_interm_dir = outputs_dir + "interm/"

special_tokens_in_vocab = ["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"]

import os
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
if not os.path.exists(outputs_model_dir):
    os.makedirs(outputs_model_dir)
if not os.path.exists(outputs_results_dir):
    os.makedirs(outputs_results_dir)
if not os.path.exists(outputs_figures_dir):
    os.makedirs(outputs_figures_dir)
if not os.path.exists(outputs_interm_dir):
    os.makedirs(outputs_interm_dir)

_global_verbose_print = True
