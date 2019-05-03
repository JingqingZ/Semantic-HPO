
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import config, dataloader, utils

def load_sentence2alpha_mapping(corpus_to_analysis, softmax=True):

    sentence2alpha = dict()

    with open(config.outputs_results_dir + "mimic_alpha_%s.npy" % corpus_to_analysis, 'rb') as f:
        mimic_alpha_results = np.load(f)
    print("MIMIC (%s) Alpha Num of Sentence: %d"% (corpus_to_analysis, mimic_alpha_results.shape[0]))

    corpus_mimic_data, corpus_hpo_data = dataloader.get_corpus()

    if corpus_to_analysis == 'train':
        train_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_train_indices]
        mimic_sentence_list = [
            sentence
            for doc in train_corpus_mimic for sentence in doc.split("\n") if len(sentence) > 0
        ]
    elif corpus_to_analysis == 'test':
        test_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_test_indices]
        mimic_sentence_list = [
            sentence
            for doc in test_corpus_mimic for sentence in doc.split("\n") if len(sentence) > 0
        ]
    else:
        raise Exception("Invalid corpus_to_analysis")

    assert len(mimic_sentence_list) == mimic_alpha_results.shape[0]

    for sidx, sent in enumerate(tqdm(mimic_sentence_list)):
        if sent in sentence2alpha:
            continue
        if softmax:
            sentence2alpha[sent] = utils.softmax(mimic_alpha_results[sidx])
        else:
            sentence2alpha[sent] = mimic_alpha_results[sidx]

    return sentence2alpha

def results_of_alpha_out(threshold, mode):

    unsupervised_method_results_file = config.outputs_results_dir + "unsupervised_method_%s_%.2f.csv" % (mode, threshold)

    assert mode == 'argmax' or mode == 'all' or mode == 'doc' or mode == 'multi'

    if not os.path.exists(unsupervised_method_results_file):
        test_sentence2alpha = load_sentence2alpha_mapping(corpus_to_analysis='test')
        train_sentence2alpha = load_sentence2alpha_mapping(corpus_to_analysis='train')

        sentence2alpha = {**train_sentence2alpha, **test_sentence2alpha}
        del train_sentence2alpha, test_sentence2alpha

        mimic_data = dataloader.load_mimic()

        def text2hpo(text):
            sentences = text.split("\n")
            hpodict = dict()
            hposet = set()
            for sent in sentences:
                if len(sent) == 0:
                    continue
                # assert sent in sentence2alpha
                if sent not in sentence2alpha:
                    continue
                if mode == 'argmax':
                    if np.max(sentence2alpha[sent]) > threshold:
                        hpoid = np.argmax(sentence2alpha[sent])
                        # hposet.add(dataloader.hpo_limited_list[hpoid])
                        hpodict[dataloader.hpo_limited_list[hpoid]] = 0
                elif mode == 'all':
                    for hpoid in range(sentence2alpha[sent].shape[0]):
                        if sentence2alpha[sent][hpoid] > threshold:
                            # hposet.add(dataloader.hpo_limited_list[hpoid])
                            hpodict[dataloader.hpo_limited_list[hpoid]] = 0
                elif mode == 'multi':
                    maxval = np.max(sentence2alpha[sent])
                    if maxval > threshold:
                        hpoid = np.argmax(sentence2alpha[sent])
                        hpodict[dataloader.hpo_limited_list[hpoid]] = 0
                    elif maxval > 0.4:
                        for hpoid in range(sentence2alpha[sent].shape[0]):
                            if sentence2alpha[sent][hpoid] / maxval > threshold:
                                hpodict[dataloader.hpo_limited_list[hpoid]] = 0
                elif mode == 'doc':
                    for hpoid in range(sentence2alpha[sent].shape[0]):
                        if sentence2alpha[sent][hpoid] < threshold:
                            continue
                        if sentence2alpha[sent][hpoid] > 0.7:
                            hposet.add(dataloader.hpo_limited_list[hpoid])
                        hpodict[dataloader.hpo_limited_list[hpoid]] = \
                            hpodict.get(dataloader.hpo_limited_list[hpoid], 0.0) + sentence2alpha[sent][hpoid]
                else:
                    raise Exception("Invalid mode")
            if mode == 'argmax' or mode == 'all' or mode == 'multi':
                hpolist = hpodict.keys()
            elif mode == 'doc':
                sorted_hpolist = sorted(hpodict.items(), key=lambda kv: kv[1])
                hpolist = [t[0] for t in sorted_hpolist][:4] + [hpo for hpo in hposet]
            else:
                raise Exception("Invalid mode")
            hpostr = "/".join(hpolist)
            return hpostr

        print("Processing alpha results ...")
        mimic_data["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"] = mimic_data["CLEAN_TEXT"].apply(text2hpo)
        mimic_data.to_csv(unsupervised_method_results_file)
        print("Saved to %s" % unsupervised_method_results_file)

    else:
        print("Loading from %s" % unsupervised_method_results_file)
        mimic_data = pd.read_csv(unsupervised_method_results_file)

    return mimic_data


if __name__ == '__main__':
    # mimic_data = results_of_alpha_out(threshold=0.5, mode='argmax')
    # threshold 0.5
    # Num of EHR has HPO 52151/52722
    # Avg HPO for all 5
    # Median HPO for all 4
    # Avg HPO for those have 5
    # Median HPO for those have 4

    # mimic_data = results_of_alpha_out(threshold=0.2, mode='argmax')
    # Num of EHR has HPO 52721/52722
    # Avg HPO for all 14
    # Median HPO for all 14
    # Avg HPO for those have 14
    # Median HPO for those have 14

    # mimic_data = results_of_alpha_out(threshold=0.2, mode='all')
    # Num of EHR has HPO 52721/52722
    # Avg HPO for all 15
    # Median HPO for all 16
    # Avg HPO for those have 15
    # Median HPO for those have 16

    # mimic_data = results_of_alpha_out(threshold=0.0, mode='doc')
    # Num of EHR has HPO 52721/52722
    # Avg HPO for all 6
    # Median HPO for all 6
    # Avg HPO for those have 6
    # Median HPO for those have 6

    mimic_data = results_of_alpha_out(threshold=0.7, mode='multi')
    # Num of EHR has HPO 52626/52722
    # Avg HPO for all 7
    # Median HPO for all 7
    # Avg HPO for those have 7
    # Median HPO for those have 7

    hpo_list = mimic_data["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"].tolist()
    print("Num of EHR has HPO %d/%d" % (np.sum([1 for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]), len(hpo_list)))
    print("Avg HPO for all %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Median HPO for all %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Avg HPO for those have %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    print("Median HPO for those have %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))

