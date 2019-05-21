
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import config, dataloader, utils, mimic_proc, evaluation

def load_sentence2alpha_mapping(corpus_to_analysis, sigmoid=True):

    sentence2alpha = dict()

    with open(config.outputs_results_dir + "mimic_alpha_%s_55000.npy" % corpus_to_analysis, 'rb') as f:
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
        # TODO: change to sigmoid
        if sigmoid:
            sentence2alpha[sent] = utils.sigmoid(mimic_alpha_results[sidx])
        else:
            sentence2alpha[sent] = mimic_alpha_results[sidx]

    return sentence2alpha

def results_of_alpha_out(threshold, mode):


    assert mode == 'argmax' or mode == 'all' or mode == 'doc' or mode == 'var_threshold' or mode == 'multi'

    if mode == 'var_threshold':
        assert len(threshold) == len(dataloader.hpo_limited_list)
        unsupervised_method_results_file = config.outputs_results_dir + "unsupervised_method_%s_%.4f.csv" % (mode, np.mean(threshold))
    else:
        unsupervised_method_results_file = config.outputs_results_dir + "unsupervised_method_%s_%.2f.csv" % (mode, threshold)

    if not os.path.exists(unsupervised_method_results_file):
    # if True:

        if mode == 'var_threshold':
            threshold_file = config.outputs_results_dir + "unsupervised_method_%s_%.4f_threshold.txt" % (mode, np.mean(threshold))
            with open(threshold_file, 'w') as thfile:
                for t in threshold:
                    thfile.write("%.4f\n" % t)

        test_sentence2alpha = load_sentence2alpha_mapping(corpus_to_analysis='test')
        train_sentence2alpha = load_sentence2alpha_mapping(corpus_to_analysis='train')
        # train_sentence2alpha = {}

        sentence2alpha = {**train_sentence2alpha, **test_sentence2alpha}
        del train_sentence2alpha, test_sentence2alpha

        '''
        for idx, sent in enumerate(sentence2alpha):
            print(sent)
            print(["%.4f" % a for a in sentence2alpha[sent]])
            if idx > 10:
                break
        exit()
        '''

        mimic_data = dataloader.load_mimic()

        # TODO: remove
        # hpo_data = dataloader.get_hpo4dataset()
        # children_info = dataloader.get_hpo_children_info()

        def text2hpo(text):
            sentences = mimic_proc.get_sentences_from_mimic(text)
            # sentences = text.split("\n")
            hpodict = dict()
            hposet = set()

            '''
            # TODO: remove
            positive_list = list()
            negative_list = list()
            for idx in range(len(dataloader.hpo_limited_list)):
                positive_list.append([])
                negative_list.append([])
            '''

            for sent in sentences:
                if len(sent) == 0:
                    continue
                # assert sent in sentence2alpha
                if sent not in sentence2alpha:
                    continue

                '''
                # TODO: remove
                print(sent)
                print(sentence2alpha[sent])
                for hidx, hpo_id in enumerate(dataloader.hpo_limited_list):
                    children = children_info[hpo_id]
                    positive_flag = False
                    for cnode in children:
                        terms = hpo_data[cnode]['terms']
                        for term in terms:
                            if term + " " in sent + " ":
                                print(sentence2alpha[sent][hidx], hidx, hpo_id, term, cnode)
                                positive_flag = True
                    if positive_flag:
                        positive_list[hidx].append(sentence2alpha[sent][hidx])
                    else:
                        negative_list[hidx].append(sentence2alpha[sent][hidx])
                '''

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
                    elif maxval > 0.3:
                        for hpoid in range(sentence2alpha[sent].shape[0]):
                            if sentence2alpha[sent][hpoid] / maxval > threshold:
                                hpodict[dataloader.hpo_limited_list[hpoid]] = 0
                elif mode == 'var_threshold':
                    for hpoid in range(sentence2alpha[sent].shape[0]):
                        if sentence2alpha[sent][hpoid] > threshold[hpoid]:
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
            if mode == 'argmax' or mode == 'all' or mode == 'multi' or mode == 'var_threshold':
                hpolist = hpodict.keys()
            elif mode == 'doc':
                sorted_hpolist = sorted(hpodict.items(), key=lambda kv: kv[1])
                hpolist = [t[0] for t in sorted_hpolist][:4] + [hpo for hpo in hposet]
            else:
                raise Exception("Invalid mode")
            hpostr = "/".join(hpolist)

            '''
            # TODO: remove
            for hidx in range(len(dataloader.hpo_limited_list)):
                print("==============")
                print(hidx, dataloader.hpo_limited_list[hidx])
                if len(positive_list[hidx]) == 0:
                    print("No positive")
                else:
                    print("%.3f %.3f %.3f" %
                          (np.max(positive_list[hidx]),
                           np.mean(positive_list[hidx]),
                           np.min(positive_list[hidx])
                    ))
                if len(negative_list[hidx]) == 0:
                    print("No negative")
                else:
                    print("%.3f %.3f %.3f" %
                          (np.max(negative_list[hidx]),
                           np.mean(negative_list[hidx]),
                           np.min(negative_list[hidx])
                           ))
            '''
            return hpostr

        print("Processing alpha results ...")

        # mimic_text = mimic_data["CLEAN_TEXT"].tolist()
        # hpostr0 = text2hpo(mimic_text[0])
        # hpostr0 = text2hpo(mimic_text[233])
        # hpostr1 = text2hpo(mimic_text[1])
        # exit()

        mimic_data["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"] = mimic_data["CLEAN_TEXT"].apply(text2hpo)
        mimic_data.to_csv(unsupervised_method_results_file)
        print("Saved to %s" % unsupervised_method_results_file)

    else:
        print("Loading from %s" % unsupervised_method_results_file)
        mimic_data = pd.read_csv(unsupervised_method_results_file)

    return mimic_data

def analyze_alpha():

    with open(config.outputs_results_dir + "mimic_alpha_%s_55000.npy" % 'train', 'rb') as f:
        mimic_alpha_train_results = np.load(f)

    new_threshold = []
    mean_std_list = []
    for i in range(len(dataloader.hpo_limited_list)):
        hpo_alpha = utils.sigmoid(mimic_alpha_train_results[:, i])
        mean = np.mean(hpo_alpha)
        std = np.std(hpo_alpha)
        normalized_hpo_alpha = (hpo_alpha - mean) / std
        # print((threshold[i] - mean) / std)
        # print(np.mean(normalized_hpo_alpha))
        # print(np.std(normalized_hpo_alpha))
        # nt = 1.28 * std + mean
        # nt = 1.04 * std + mean
        nt = 0.84 * std + mean
        # nt = 0.53 * std + mean
        new_threshold.append(nt)
        mean_std_list.append((mean, std))
        # print("----")

    if config._global_verbose_print:
        print(", ".join(["%.3f" % nt for nt in new_threshold]) )
    return new_threshold, mean_std_list

def results_of_alpha_out_norm_topk(k):

    unsupervised_method_results_file = config.outputs_results_dir + "unsupervised_method_norm_%d.csv" % (k)

    # if not os.path.exists(unsupervised_method_results_file):
    if True:

        test_sentence2alpha = load_sentence2alpha_mapping(corpus_to_analysis='test')
        train_sentence2alpha = load_sentence2alpha_mapping(corpus_to_analysis='train')
        # train_sentence2alpha = {}

        sentence2alpha = {**train_sentence2alpha, **test_sentence2alpha}
        del train_sentence2alpha, test_sentence2alpha

        mimic_data = dataloader.load_mimic()

        threshold, mean_std_list = analyze_alpha()

        threshold=[
            0.3, 0.35, 0.08, 0.7,
            0.4, 0.7, 0.25, 0.1,
            0.85, 0.4, 0.2, 0.03,
            0.01, 0.7, 0.4, 0.02,
            0.85, 0.05, 0.4, 0.8,
            0.15, 0.55, 0.65, 0.6
        ]

        def text2hpo(text):
            sentences = mimic_proc.get_sentences_from_mimic(text)

            all_alpha_out = list()
            all_candidate_hidx = set()
            for sent in sentences:
                if len(sent) == 0:
                    continue
                # assert sent in sentence2alpha
                if sent not in sentence2alpha:
                    continue

                alpha_out = sentence2alpha[sent]
                candidate_hidx = set()

                for i in range(alpha_out.shape[0]):
                    if alpha_out[i] > threshold[i]:
                        candidate_hidx.add(i)
                    alpha_out[i] = (alpha_out[i] - mean_std_list[i][0]) / mean_std_list[i][1]

                all_alpha_out.append(alpha_out)
                all_candidate_hidx.update(candidate_hidx)

            if len(all_candidate_hidx) == 0:
                return ""

            all_alpha_out = np.stack(all_alpha_out, axis=0)
            reduce_alpha_out = np.max(all_alpha_out, axis=0)

            arghidx = np.argsort(reduce_alpha_out)[::-1]
            # arghidx = set(arghidx)

            final_hidx = []
            for i in range(len(dataloader.hpo_limited_list)):
                if len(final_hidx) >= k:
                    break
                if arghidx[i] in all_candidate_hidx:
                    final_hidx.append(arghidx[i])

            # final_hidx = arghidx & all_candidate_hidx
            final_hidx = all_candidate_hidx
            hpolist = [dataloader.hpo_limited_list[hidx] for hidx in final_hidx]

            hpostr = "/".join(hpolist)

            return hpostr

        if config._global_verbose_print:
            print("Processing alpha results ...")


        if config._global_verbose_print:
            tqdm.pandas()
            mimic_data["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"] = mimic_data["CLEAN_TEXT"].progress_apply(text2hpo)
        else:
            mimic_data["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"] = mimic_data["CLEAN_TEXT"].apply(text2hpo)
        mimic_data.to_csv(unsupervised_method_results_file)
        if config._global_verbose_print:
            print("Saved to %s" % unsupervised_method_results_file)

    else:
        print("Loading from %s" % unsupervised_method_results_file)
        mimic_data = pd.read_csv(unsupervised_method_results_file)

    return mimic_data

def evaluate_results_of_alpha_out_norm_topk_iteration():

    test_sentence2alpha = load_sentence2alpha_mapping(corpus_to_analysis='test')
    train_sentence2alpha = load_sentence2alpha_mapping(corpus_to_analysis='train')
    # train_sentence2alpha = {}

    sentence2alpha = {**train_sentence2alpha, **test_sentence2alpha}
    del train_sentence2alpha, test_sentence2alpha

    threshold, mean_std_list = analyze_alpha()

    threshold=[
        0.3, 0.35, 0.08, 0.7,
        0.4, 0.7, 0.25, 0.1,
        0.85, 0.4, 0.2, 0.03,
        0.01, 0.7, 0.4, 0.02,
        0.85, 0.05, 0.4, 0.8,
        0.15, 0.55, 0.65, 0.6
    ]

    mimic_data = dataloader.load_mimic()
    silver = evaluation.get_icd2hpo_3digit_results()

    # for k in range(len(dataloader.hpo_limited_list)):
    for k in range(1, 14):

        def text2hpo(text):
            sentences = mimic_proc.get_sentences_from_mimic(text)

            all_alpha_out = list()
            all_candidate_hidx = set()
            for sent in sentences:
                if len(sent) == 0:
                    continue
                # assert sent in sentence2alpha
                if sent not in sentence2alpha:
                    continue

                alpha_out = sentence2alpha[sent]
                candidate_hidx = set()

                for i in range(alpha_out.shape[0]):
                    if alpha_out[i] > threshold[i]:
                        candidate_hidx.add(i)
                    alpha_out[i] = (alpha_out[i] - mean_std_list[i][0]) / mean_std_list[i][1]

                all_alpha_out.append(alpha_out)
                all_candidate_hidx.update(candidate_hidx)

            if len(all_candidate_hidx) == 0:
                return ""

            all_alpha_out = np.stack(all_alpha_out, axis=0)
            reduce_alpha_out = np.max(all_alpha_out, axis=0)

            arghidx = np.argsort(reduce_alpha_out)[::-1]
            # arghidx = set(arghidx)

            final_hidx = []
            for i in range(len(dataloader.hpo_limited_list)):
                if len(final_hidx) >= k:
                    break
                if arghidx[i] in all_candidate_hidx:
                    final_hidx.append(arghidx[i])

            # final_hidx = arghidx & all_candidate_hidx
            hpolist = [dataloader.hpo_limited_list[hidx] for hidx in final_hidx]

            hpostr = "/".join(hpolist)

            return hpostr

        series = mimic_data["CLEAN_TEXT"].apply(text2hpo)
        unsuper = series.tolist()

        print("Avg of HPO / EHR: %d " % (k))
        evaluation._evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [unsuper[index] for index in config.mimic_test_indices],
            func=evaluation.jaccard
        )
        evaluation._evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [unsuper[index] for index in config.mimic_test_indices],
            func=evaluation.overlap_coefficient
        )

def annotation_with_threshold():

    threshold=[
        0.3, 0.35, 0.08, 0.7,
        0.4, 0.7, 0.25, 0.1,
        0.85, 0.4, 0.2, 0.03,
        0.01, 0.7, 0.4, 0.02,
        0.85, 0.05, 0.4, 0.8,
        0.15, 0.55, 0.65, 0.6
    ]

    column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"

    decision_mode = 'var_threshold'
    unsuper = results_of_alpha_out(threshold=threshold, mode=decision_mode)[column_of_keyword].tolist()
    return unsuper

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

    # mimic_data = results_of_alpha_out(threshold=0.7, mode='multi')
    # Num of EHR has HPO 52626/52722
    # Avg HPO for all 7
    # Median HPO for all 7
    # Avg HPO for those have 7
    # Median HPO for those have 7

    # TODO: analyze results
    # why hpo no 5 prob is so high

    # mimic_data = results_of_alpha_out(threshold=0.1, mode='all')
    # mimic_data = results_of_alpha_out(threshold=0.4, mode='all')
    # mimic_data = results_of_alpha_out_norm_topk(8)
    # hpo_list = mimic_data["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"].tolist()
    # hpo_list = evaluation.get_icd2hpo_3digit_results()
    # utils.print_statictis_of_hpo(hpo_list)
    config._global_verbose_print = False
    evaluate_results_of_alpha_out_norm_topk_iteration()
