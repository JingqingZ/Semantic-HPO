
import re
import random
import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import Dataset

import config
from log import logger
import dataloader

# The code is modified and rewritten based on
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_lm_finetuning.py
class BERTDataset(Dataset):
    def __init__(self, corpus_data, tokenizer, hpo_data=None, hpo_ontology=None, seq_len=config.sequence_length):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.sample_to_doc = [] # map sample index to doc and line

        self.all_docs = []
        print("Initializing the MIMIC dataset for training ...")
        for doc in tqdm(corpus_data):
            if len(doc) == 0:
                continue
            lines = doc.split("\n")
            tdoc = []
            for line in lines:
                if len(line) < config.sequence_length // 10:
                    continue
                sample = {
                    "doc_id": len(self.all_docs),
                    "line_id": len(tdoc)
                }
                tdoc.append(line)
                self.sample_to_doc.append(sample)
            # the last sentence of a doc should be removed as there wont be a subsequent sentence anymore
            self.sample_to_doc.pop()
            assert len(tdoc) > 0
            self.all_docs.append(tdoc)

        self.sample_of_hpo = []
        if hpo_data is not None and hpo_ontology is not None:
            print("Initializing the HPO dataset for training ...")

            self.hpo_data = hpo_data
            self.hpo_all_keys = list(hpo_data.keys())
            self.hpo_connection_mapping = dict()

            t_mapping = dict()
            def dfs(node):
                if node in t_mapping:
                    return
                children = hpo_ontology[node]["relations"].get("can_be", [])
                t_mapping[node] = set(children)
                for child in children:
                    dfs(child)

            dfs(dataloader.hpo_root_id)

            for node in t_mapping:
                if len(t_mapping[node]) > 0:
                    self.hpo_connection_mapping[node] = t_mapping[node]

            for node in tqdm(self.hpo_connection_mapping):
                for child in self.hpo_connection_mapping[node]:
                    sample = (node, child)
                    self.sample_of_hpo.append(sample)

        self.num_samples = len(self.sample_to_doc) + len(self.sample_of_hpo)
        self.num_docs = len(self.all_docs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):

        if item < len(self.sample_to_doc):
            t1, t2, is_next_label = self.random_sent(item)
        else:
            t1, t2, is_next_label = self.random_hpo(item - len(self.sample_to_doc))

        cur_id = self.sample_counter
        self.sample_counter += 1

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        tokens_b = self.tokenizer.tokenize(t2)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next))

        return cur_tensors


    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, label

    def random_hpo(self, index):
        """
        Get one sample consisting of two descriptions of two HPO terms. With prob. 50% these are two subsequent terms in the hierarchy,
        which means they has a parent-child relation.
        With 50% the second HPO will be a random one from another branch in the hierarchy.
        :param index: int, index of sample.
        :return: (str, str, int), HPO description 1, HPO description 2, isNextSentence Label
        """
        assert index < len(self.sample_of_hpo)

        hpo_t1, hpo_t2 = self.sample_of_hpo[index]

        if random.random() > 0.5:
            label = 0
        else:
            for _ in range(100):
                random_key = random.choice(self.hpo_all_keys)
                if random_key not in self.hpo_connection_mapping[hpo_t1]:
                    break
                if _ == 99:
                    raise Exception("Cannot randomly pick a HPO term for %s" % hpo_t1)
            hpo_t2 = random_key
            label = 1

        assert hpo_t1 in self.hpo_data
        assert hpo_t2 in self.hpo_data

        desc1 = self.hpo_data[hpo_t1]
        desc2 = self.hpo_data[hpo_t2]

        return desc1, desc2, label

    def get_corpus_line(self, index):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert index < len(self.sample_to_doc)

        sample = self.sample_to_doc[index]
        t1 = self.all_docs[sample["doc_id"]][sample["line_id"]]
        t2 = self.all_docs[sample["doc_id"]][sample["line_id"] + 1]
        # used later to avoid random nextSentence from same doc
        self.current_doc = sample["doc_id"]

        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.

        for _ in range(100):
            rand_doc_idx = random.randint(0, len(self.all_docs)-1)
            rand_doc = self.all_docs[rand_doc_idx]
            line = rand_doc[random.randrange(len(rand_doc))]
            #check if our picked random line is really from another doc like we want it to be
            if rand_doc_idx != self.current_doc:
                return line
        raise Exception("Get random line error.")

class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids

def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_b, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    '''
    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))
        logger.info("Is next sentence label: %s " % (example.is_next))
    '''

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next)
    return features

def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class HPOAnnotate4TrainingDataset(Dataset):
    def __init__(self, hpodata, tokenizer, seq_len=config.sequence_length):

        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.hpodata = hpodata

        self.sentence_dict = dict()
        self.sentence_list = list()

        self.hpo_dict = dict()
        self.hpo_list = list()

        # for training only
        # create a complete index for all sentences
        for hpoid in self.hpodata:
            for sentence in self.hpodata[hpoid]['mimic_train']:
                if sentence in self.sentence_dict:
                    sentence_id = self.sentence_dict[sentence]
                    self.sentence_list[sentence_id]['hpo_ids'].add(hpoid)
                    continue
                assert sentence.replace("[UNK]", "").islower()
                assert len(sentence) > 0
                assert sentence not in self.sentence_dict
                self.sentence_dict[sentence] = len(self.sentence_list)
                self.sentence_list.append({
                    "sentence": sentence,
                    "hpo_ids": {hpoid}
                })
            description = self.hpodata[hpoid]['description']
            # description = self.hpodata[hpoid]['description'].strip().lower()
            # regex = re.compile('[^a-z\s]')
            # description = regex.sub(" ", description)
            # self.hpodata[hpoid]['description'] = description
            assert description.replace("[UNK]", "").islower()
            assert description not in self.sentence_dict
            self.sentence_dict[description] = len(self.sentence_list)
            self.sentence_list.append({
                "sentence": description,
                "hpo_ids": {hpoid}
            })

        print("Processing training dataset for HPO annotation ...")
        # perform DFS to construct relations between HPO and sentences (HPO -> sentences)
        self.dfs(dataloader.hpo_root_id, depth=0)

        del self.hpodata
        # in case anything needs to be changed
        # with open(dataloader.hpo_dataset_file, 'wb') as f:
        #     pickle.dump(self.hpodata, f)
        # exit()

        # construct relations sentences -> HPO
        for hpo_sample in tqdm(self.hpo_list):
            hpo_id = hpo_sample['hpo_id']
            for sentence_id in hpo_sample['sentences_id']:
                self.sentence_list[sentence_id]["hpo_ids"].add(hpo_id)

        # no sentence should be related to all HPO terms
        # every sentence should be related to at least one HPO term
        for sentence_sample in self.sentence_list:
            assert len(sentence_sample["hpo_ids"]) < len(self.hpo_dict)
            assert len(sentence_sample["hpo_ids"]) > 0

        assert len(self.hpo_list) == len(self.hpo_dict)
        assert len(self.sentence_list) == len(self.sentence_dict)
        print("Number of HPO terms: %d" % len(self.hpo_list))
        print("Avg number of sentences per HPO: %f" % (np.mean([len(sample["sentences_id"]) for sample in self.hpo_list])))
        print("Median number of sentences per HPO: %f" % (np.median([len(sample["sentences_id"]) for sample in self.hpo_list])))
        print("Max number of HPO per sentence: %f" % (np.max([len(sample["hpo_ids"]) for sample in self.sentence_list])))
        print("Avg number of HPO per sentence: %f" % (np.mean([len(sample["hpo_ids"]) for sample in self.sentence_list])))
        print("Median number of HPO per sentence: %f" % (np.median([len(sample["hpo_ids"]) for sample in self.sentence_list])))
        print("Total number of related sentences: %d" % len(self.sentence_list))
        # Number of HPO terms: 13993
        # Avg number of sentences per HPO: 1215.436147
        # Median number of sentences per HPO: 1.000000
        # Max number of HPO per sentence: 93.000000
        # Avg number of HPO per sentence: 12.020353
        # Median number of HPO per sentence: 10.000000
        # Total number of related sentences: 1414900

    def dfs(self, hpoid, depth):

        if hpoid in self.hpo_dict:
            return

        # assert 'depth' in self.hpodata[hpoid]
        # self.hpodata[hpoid]['depth'] = depth

        if self.hpodata[hpoid]['status'] == True:
            assert hpoid not in self.hpo_dict
            self.hpo_dict[hpoid] = len(self.hpo_list)
            self.hpo_list.append({
                "hpo_id": hpoid,
                "description_id": self.sentence_dict[self.hpodata[hpoid]['description']],
                "sentences_id": set([self.sentence_dict[s] for s in self.hpodata[hpoid]['mimic_train']]) # only train data will be included
            })
            self.hpo_list[-1]["sentences_id"].add(self.hpo_list[-1]["description_id"])
            # sentences of all children nodes should be included as well
            for child in self.hpodata[hpoid]['children_node']:
                self.dfs(child, depth + 1)
                self.hpo_list[self.hpo_dict[hpoid]]["sentences_id"] |= self.hpo_list[self.hpo_dict[child]]["sentences_id"]
        else:
            for child in self.hpodata[hpoid]['children_node']:
                self.dfs(child, depth + 1)

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, item):

        if random.random() > 0.5:
            # positive sample
            hpoid = random.choice(list(self.sentence_list[item]['hpo_ids']))
            hpo_desc_id = self.hpo_list[self.hpo_dict[hpoid]]['description_id']
            is_related = 1
        else:
            # negative sample
            while True:
                hpoid = random.choice(list(self.hpo_dict.keys()))
                if hpoid not in self.sentence_list[item]['hpo_ids']:
                    hpo_desc_id = self.hpo_list[self.hpo_dict[hpoid]]['description_id']
                    break
            is_related = 0

        sentence = self.sentence_list[item]['sentence']
        hpo_desc = self.sentence_list[hpo_desc_id]['sentence']

        tokens_a = self.tokenizer.tokenize(sentence)
        tokens_b = self.tokenizer.tokenize(hpo_desc)

        # combine to one sample
        cur_example = InputExample4HPOAnnotation(tokens_a=tokens_a, tokens_b=tokens_b, is_related=is_related)

        # transform sample to features
        cur_features = convert_example_to_features_4_hpo_annotation(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.is_related))

        return cur_tensors

class InputExample4HPOAnnotation(object):
    """A single training/test example for the language model."""

    def __init__(self, tokens_a, tokens_b, is_related):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            tokens_b: string. The untokenized text of the second sequence.
        """
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_related = is_related

class InputFeatures4HPOAnnotation(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_related):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_related = is_related

def convert_example_to_features_4_hpo_annotation(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = InputFeatures4HPOAnnotation(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_related=example.is_related)
    return features

class SentenceEmbeddingDataset(Dataset):
    def __init__(self, sentences_list, tokenizer, seq_len=config.sequence_length):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.samples = sentences_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        text = self.samples[item]
        tokens_a = self.tokenizer.tokenize(text)

        while True:
            if len(tokens_a) <= self.seq_len - 2:
                break
            else:
                tokens_a.pop()

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.seq_len
        assert len(input_mask) == self.seq_len
        assert len(segment_ids) == self.seq_len

        cur_tensors = (torch.tensor(input_ids),
                       torch.tensor(input_mask),
                       torch.tensor(segment_ids))

        return cur_tensors




if __name__ == '__main__':
    pass









