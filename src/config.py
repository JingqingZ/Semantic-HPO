
train_epoch = 10
train_batch_size = 64
test_batch_size = 64

save_step_4_pretrain = 5000
save_step_4_annotation = 5000
print_step = 100

sequence_length = 64
vocabulary_size = 30000

learning_rate_4_pretrain = 3e-4
learning_rate_4_annotation = 1e-3
warmup_proportion = 0.1 # proportion of training to perform linear learning rate warmup for.

step_to_train_all_weights = 20000

# TODO: MIMIC data are not shuffled, not sure if shuffle is necessary, seems not
training_percentage = 0.7
testing_percentage = 0.3
assert training_percentage + testing_percentage == 1

outputs_dir = "../outputs/"
outputs_model_dir = outputs_dir + "models/"

special_tokens_in_vocab = ["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"]

import os
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
if not os.path.exists(outputs_model_dir):
    os.makedirs(outputs_model_dir)


## Some statictis:
# Total num of MIMIC docs: 52722
# Total num of MIMIC sentences: 483598154
# Number of HPO terms: 13993
# Avg number of sentences per HPO: 1215.436147
# Median number of sentences per HPO: 1.000000
# Max number of HPO per sentence: 93.000000
# Avg number of HPO per sentence: 12.020353
# Median number of HPO per sentence: 10.000000
# Total number of related sentences: 1414900
