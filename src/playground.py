## The code in this file is used for analysis (or playground)
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

import utils
import config
import dataloader
import baselines
import evaluation
import decision

def orphadata():
    data_folder = "../data/orphadata/"
    hpo_file = data_folder + "en_product4_HPO.xml"

    import xml.etree.ElementTree as ET
    tree = ET.parse(hpo_file)
    root = tree.getroot()

def icd_distribution_in_mimic():
    from collections import Counter

    mimic_data = dataloader.load_mimic()
    full_icd_list = mimic_data['ICD9_CODE_LIST'].tolist()

    icd2hpo = dataloader.get_icd_hpo_silver_mapping()

    def _icd_dist(icd_list):
        icd_counter = Counter()
        icd_has_hpo_counter = Counter()
        for icdstr in icd_list:
            icds = icdstr.split("/")
            for icd in icds:
                icd_counter[icd] += 1
                if icd in icd2hpo:
                    icd_has_hpo_counter[icd] += 1
        return icd_counter, icd_has_hpo_counter

    icd_train_counter, icd_has_hpo_train_counter = _icd_dist([full_icd_list[index] for index in config.mimic_train_indices])
    icd_test_counter, icd_has_hpo_test_counter = _icd_dist([full_icd_list[index] for index in config.mimic_test_indices])
    print(icd_train_counter.most_common(50))
    print(icd_test_counter.most_common(50))
    print(icd_has_hpo_train_counter.most_common(50))
    print(icd_has_hpo_test_counter.most_common(50))

def num_mimic_negation():
    mimic_data = dataloader.load_mimic()
    mimic_doc = mimic_data['CLEAN_TEXT'].tolist()
    negative_word = ['not', 'never', 'hardly', 'rarely', 'rule out',
                     'ruled out', 'no', 'no evidence', 'no support', 'no suspicion',
                     'deny', 'denies', 'denied', 'unlikely', 'no change', 'without',
                     'no signs of', 'absence', 'not demonstrated', 'negative', 'doubt',
                     'versus', 'negation', 'no further', 'without any further', 'without further']
    negative_word = [' ' + w + ' ' for w in negative_word]
    negative_num = [0 for _ in negative_word]
    for doc in tqdm(mimic_doc):
        for widx, nword in enumerate(negative_word):
            negative_num[widx] += 1 if nword in doc else 0
    print("Total doc %d" % len(mimic_doc))
    print("Negation %s" % ", ".join(["%s:%d" % (nword, negative_num[widx]) for widx, nword in enumerate(negative_word)]))
    # Total doc 52722
    # Negation  not :44867,  never :6730,  hardly :31,  rarely :803,  rule out :5183,  ruled out :5201,  no :52134,
    # no evidence :19646,  no support :6,  no suspicion :30,  deny :210,  denies :22140,  denied :7377,  unlikely :3105,
    # no change :2930,  without :34670,  no signs of :2859,  absence :3117,  not demonstrated :64,  negative :31487,
    # doubt :170,  versus :4238,  negation :0,  no further :6168,  without any further :212,  without further :660

def avg_hpo_per_sentence():
    mimic_data, _ = dataloader.get_corpus()

    counter = 0
    for doc in mimic_data:
        if len(doc) == 0:
            continue
        counter += len(doc.split("\n"))

    hpo_data = dataloader.get_hpo4dataset()
    sentence_dict = dict()

    for node in hpo_data:
        for sentence in (hpo_data[node]['mimic_train'] | hpo_data[node]['mimic_test']):
            sentence_dict[sentence] = sentence_dict.get(sentence, 0) + 1

    print("Total num of sentence that have HPO: %d / %d" % (len(sentence_dict), counter))
    print("Max num of HPO per sentence: %.f" % np.max([sentence_dict[s] for s in sentence_dict]))
    print("Q90 num of HPO per sentence: %.f" % np.quantile([sentence_dict[s] for s in sentence_dict], 0.9))
    print("Avg num of HPO per sentence: %.f" % np.mean([sentence_dict[s] for s in sentence_dict]))
    print("Median num of HPO per sentence: %.f" % np.median([sentence_dict[s] for s in sentence_dict]))
    # Total num of sentence that have HPO: 1998024 / 2575124
    # Max num of HPO per sentence: 23
    # Q90 num of HPO per sentence: 6
    # Avg num of HPO per sentence: 3
    # Median num of HPO per sentence: 2

def avg_hpo_per_sentence2():
    # only analysis the target hpo terms
    mimic_data, _ = dataloader.get_corpus()

    counter = 0
    for doc in mimic_data:
        if len(doc) == 0:
            continue
        counter += len(doc.split("\n"))

    hpo_data = dataloader.get_hpo4dataset()
    sentence_dict = dict()

    children_info = dataloader.get_hpo_children_info()

    children_to_predecessor = dict()
    for hpoid in dataloader.hpo_limited_list:
        for cnode in children_info[hpoid]:
            if cnode not in children_to_predecessor:
                children_to_predecessor[cnode] = set()
            children_to_predecessor[cnode].add(hpoid)

    for node in children_to_predecessor:
        for sentence in (hpo_data[node]['mimic_train'] | hpo_data[node]['mimic_test']):
            if sentence not in sentence_dict:
                sentence_dict[sentence] = set()
            sentence_dict[sentence].update(children_to_predecessor[node])

    print("Total num of sentence that have HPO: %d / %d" % (len(sentence_dict), counter))
    print("Max num of HPO per sentence: %.f" % np.max([len(sentence_dict[s]) for s in sentence_dict]))
    print("Q90 num of HPO per sentence: %.f" % np.quantile([len(sentence_dict[s]) for s in sentence_dict], 0.9))
    print("Avg num of HPO per sentence: %.f" % np.mean([len(sentence_dict[s]) for s in sentence_dict]))
    print("Median num of HPO per sentence: %.f" % np.median([len(sentence_dict[s]) for s in sentence_dict]))
    # Total num of sentence that have HPO: 1842715 / 2575124
    # Max num of HPO per sentence: 13
    # Q90 num of HPO per sentence: 4
    # Avg num of HPO per sentence: 2
    # Median num of HPO per sentence: 2

    hpo_onto = dataloader.load_hpo_ontology()

    for sentence in sentence_dict:
        if len(sentence_dict[sentence]) > 10:
            print(sentence)
            hpolist = list(sentence_dict[sentence])
            print(hpolist)
            print([hpo_onto[hpo]['name'] for hpo in hpolist])
            print("---")

def analysis_overlapping_of_children_node():

    children_info = dataloader.get_hpo_children_info()

    children_to_predecessor = dict()
    for hpoid in dataloader.hpo_limited_list:
        for cnode in children_info[hpoid]:
            if cnode not in children_to_predecessor:
                children_to_predecessor[cnode] = set()
            children_to_predecessor[cnode].add(hpoid)

    counter = [0] * 5
    for cnode in children_to_predecessor:
        if len(children_to_predecessor[cnode]) >= 5:
            print(cnode)
            # HP:0005508
            # HP:0009711
            # HP:0200151
            counter[4] += 1
        else:
            counter[len(children_to_predecessor[cnode]) - 1] += 1

    print("Overlapping of children node for the target HPOs: ", counter)
    # Overlapping of children node for the target HPOs:  [8534, 4722, 453, 83, 3]

    for hpoid in dataloader.hpo_limited_list:
        print(hpoid, len(children_info[hpoid]))

def hpo_have_most_sentences():
    hpo_data = dataloader.get_hpo4dataset()

    hpo_sentence_counter = dict()

    for node in hpo_data:
        hpo_sentence_counter[node] = len(hpo_data[node]['mimic_train'] | hpo_data[node]['mimic_test'])

    sorted_list = sorted(hpo_sentence_counter.items(), key=lambda k: k[1], reverse=True)
    print("Num of HPO:", len(sorted_list))
    print(sorted_list[:20])
    print(sorted_list[-20:])
    # Num of HPO: 13994
    # [('HP:0001658', 1087259), ('HP:0500001', 281987), ('HP:0012835', 232469), ('HP:0012834', 216343), ('HP:0012531', 174305), ('HP:0002326', 162987), ('HP:0025275', 118389), ('HP:0031915', 112217), ('HP:0012825', 100589), ('HP:0012832', 93602), ('HP:0011009', 83064), ('HP:0000969', 76257), ('HP:0100021', 70951), ('HP:0012830', 63036), ('HP:0001945', 61077), ('HP:0000822', 60011), ('HP:0011010', 59163), ('HP:0002098', 50625), ('HP:0012826', 50255), ('HP:0012828', 44604)]
    # [('HP:0003365', 0), ('HP:0009856', 0), ('HP:0010126', 0), ('HP:0008080', 0), ('HP:0008946', 0), ('HP:0009460', 0), ('HP:0000117', 0), ('HP:0000433', 0), ('HP:0003145', 0), ('HP:0031734', 0), ('HP:0031591', 0), ('HP:0009613', 0), ('HP:0025098', 0), ('HP:0040023', 0), ('HP:0003863', 0), ('HP:0012381', 0), ('HP:0012881', 0), ('HP:0100657', 0), ('HP:0007992', 0), ('HP:0030758', 0)]

def analysis_sentence_embedding():

    from tsne import tsne
    import pylab

    sentence_dict = dict()
    with open(config.outputs_results_dir + "mimic_sentence_test.pickle", 'rb') as f:
        mimic_sentence_list = pickle.load(f)
    for idx, sentence in enumerate(mimic_sentence_list):
        sentence_dict[sentence] = idx
    with open(config.outputs_results_dir + "mimic_embedding_test.npy", 'rb') as f:
        mimic_embedding_list = np.load(f)
    assert len(mimic_sentence_list) == mimic_embedding_list.shape[0]

    with open(config.outputs_results_dir + "mimic_embedding_hpo.npy", 'rb') as f:
        hpo_embedding_list = np.load(f)

    X = list()
    labels = list()

    hpo_data = dataloader.get_hpo4dataset()

    for nidx, node in enumerate(hpo_data):
        if nidx >= 100:
            break
        sentences = hpo_data[node]['mimic_test']
        for s in sentences:
            ns = " ".join([w.strip() for w in s.strip().split() if len(w.strip()) > 0])
            idx = sentence_dict[ns]
            vec = mimic_embedding_list[idx]
            X.append(vec)
            labels.append(nidx)
    X = np.array(X)
    labels = np.array(labels)

    Y = tsne(X, 2, 50, 25.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 10, labels, marker='o', alpha=0.7)
    pylab.show()

def analysis_hpo_embedding(level=1, root_node=dataloader.hpo_root_id):
    from tsne import tsne
    import pylab

    with open(config.outputs_results_dir + "mimic_embedding_hpo.npy", 'rb') as f:
        hpo_embedding_list = np.load(f)
    with open(config.outputs_results_dir + "mimic_sentence_hpo.pickle", 'rb') as f:
        hpo_sentence_list = pickle.load(f)

    assert len(hpo_sentence_list) == hpo_embedding_list.shape[0]

    sentence_dict = dict()
    for i in range(len(hpo_sentence_list)):
        sentence = hpo_sentence_list[i]
        sentence_dict[sentence] = hpo_embedding_list[i]

    hpo_data = dataloader.get_hpo4dataset()
    hpo_onto = dataloader.load_hpo_ontology()

    def get_seed_node(level):
        assert level > 0 and isinstance(level, int)
        seed_node = hpo_onto[root_node]["relations"].get("can_be", [])
        for i in range(level - 1):
            seed_node = [n for node in seed_node for n in hpo_onto[node]["relations"].get("can_be", [])]
        return seed_node

    seed_node = get_seed_node(level=level)
    # print(level1_node)
    # print(level2_node)
    # desc = " ".join([w.strip() for w in hpo_data[dataloader.hpo_root_id]["description"].strip().split() if len(w.strip()) > 0])

    print(seed_node, len(seed_node))
    for s in seed_node:
        print(s, hpo_data[s]["description"])
    X = list()
    labels = list()
    for idx, snode in enumerate(seed_node):
        cnodes = hpo_data[snode]["children_node"]
        for cnode in sorted(cnodes)[:20]:
            desc = " ".join([w.strip() for w in hpo_data[cnode]["description"].strip().split() if len(w.strip()) > 0])
            vec = sentence_dict[desc]
            X.append(vec)
            labels.append(idx)
    X = np.array(X)
    labels = np.array(labels)

    Y = tsne(X, 2, 50, 15.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 10, labels, marker='o', cmap='tab20c' if len(seed_node) <= 20 else 'gist_ncar')
    pylab.show()

def get_icd2hpo_in_limited_hpo_set(root_node):
    hpo_data = dataloader.get_hpo4dataset()
    hpo_onto = dataloader.load_hpo_ontology()
    seed_node = hpo_onto[root_node]["relations"].get("can_be", [])

    node_mapping = dict()
    for s in seed_node:
        successors = hpo_data[s]["children_node"]
        for c in successors:
            # assert c not in node_mapping
            if c not in node_mapping:
                node_mapping[c] = set()
            node_mapping[c].add(s)

    icd2hpo = dataloader.get_icd_hpo_silver_mapping()
    new_icd2hpo = dict()

    for icd in icd2hpo:
        new_icd2hpo[icd] = set()
        for hpo in icd2hpo[icd]:
            if hpo in node_mapping:
                new_icd2hpo[icd].update(node_mapping[hpo])

    return new_icd2hpo

def analysis_alpha_out(corpus_to_analysis):
    import decision

    if corpus_to_analysis == 'test' or corpus_to_analysis == 'all':
        test_sentence2alpha = decision.load_sentence2alpha_mapping(corpus_to_analysis='test')
    else:
        test_sentence2alpha = {}
    if corpus_to_analysis == 'train' or corpus_to_analysis == 'all':
        train_sentence2alpha = decision.load_sentence2alpha_mapping(corpus_to_analysis='train')
    else:
        train_sentence2alpha = {}

    sentence2alpha = {**train_sentence2alpha, **test_sentence2alpha}
    del train_sentence2alpha, test_sentence2alpha

    numprint = 0
    for sent in sentence2alpha:
        if np.max(sentence2alpha[sent]) > 0.5:
            print(sent)
            print(dataloader.hpo_limited_list[np.argmax(sentence2alpha[sent])])
            print(sentence2alpha[sent])
            print("-----------")
            numprint += 1
        if numprint > 10:
            break

def description_of_hpo():
    _, corpus_hpo = dataloader.get_corpus()
    for hpo in dataloader.hpo_limited_list:
        print(hpo)
        print(corpus_hpo[hpo])
        print("----")

def distribution_of_hpo_in_silver_standard(result_list):
    hpo_counter = dict()
    for hpostr in result_list:
        if isinstance(hpostr, float):
            continue
        hpolist = hpostr.split("/")
        for hpo in hpolist:
            hpo_counter[hpo] = hpo_counter.get(hpo, 0) + 1

    for hpo in dataloader.hpo_limited_list:
        print(hpo, hpo_counter.get(hpo, 0))

    pass

def hpo_description_optimization():
    hpo_onto = dataloader.load_hpo_ontology()
    _, corpus_hpo_data = dataloader.get_corpus()

    for hpo in dataloader.hpo_limited_list:
        children = hpo_onto[hpo]["relations"].get("can_be", [])
        if len(corpus_hpo_data[hpo].split()) < config.sequence_length // 2:
            origin_word_set = set(["abnormality", "of"])
            newwordlist = list()
            for cnode in children:
                for word in corpus_hpo_data[cnode].split():
                    if word not in origin_word_set:
                        newwordlist.append(word)
            new_desc = "%s %s" % (
                corpus_hpo_data[hpo],
                " ".join(newwordlist)
            )
        else:
            new_desc = corpus_hpo_data[hpo]
        print(corpus_hpo_data[hpo])
        print(new_desc)
        print("===")


_record_idx = 1

def convert_mimic_to_plain_text():
    mimic_corpus, _ = dataloader.get_corpus()
    with open(config.outputs_interm_dir + "record%d.txt" % _record_idx, 'w') as f:
        f.write(mimic_corpus[_record_idx].replace("\n", " "))

def analyze_ehr_phenolyzer_results():
    children_info = dataloader.get_hpo_children_info()

    silver_data = baselines.silver_standard()
    print(silver_data['CLEAN_TEXT'].tolist()[0])
    print(silver_data['ICD9_CODE_LIST'].tolist()[0])

    with open("../baselines/EHR-Phenolyzer/out/record%d.HPO.txt" % _record_idx) as f:
        results_hpo = set()
        for line in f:
            hpo_id = line.split("\t")[0]
            for hpo_node in dataloader.hpo_limited_list:
                if hpo_id == hpo_node:
                    results_hpo.add(hpo_node)
                elif hpo_id in children_info[hpo_node]:
                    results_hpo.add(hpo_node)
        print(sorted(results_hpo))

    silver = silver_data['HPO_CODE_LIST'].tolist()
    silver = set(silver[_record_idx].split("/"))
    print(sorted(silver))

    column_of_keyword = 'HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'
    keyword = baselines.keyword_search()[column_of_keyword].tolist()
    keyword = set(keyword[_record_idx].split("/"))
    print(sorted(keyword))

    import evaluation
    print(evaluation.jaccard(silver, results_hpo))
    print(evaluation.jaccard(keyword, results_hpo))


def analyze_specific_case(doc_id):

    silver_data = baselines.silver_standard()
    keyword_data = baselines.keyword_search()

    icd = silver_data['ICD9_CODE_LIST'].tolist()[doc_id].split("/")
    silver = silver_data['HPO_CODE_LIST'].tolist()[doc_id].split("/")
    keyword = keyword_data['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'].tolist()[doc_id].split("/")
    text = silver_data['CLEAN_TEXT'].tolist()[doc_id]
    assert text == keyword_data['CLEAN_TEXT'].tolist()[doc_id]
    original_text = dataloader.load_mimic()['TEXT'].tolist()[doc_id]

    icd2hpo = dataloader.get_icd_hpo_silver_mapping()
    icd2hpo_predcessor = dataloader.get_icd_hpo_in_limited_hpo_set()

    import evaluation
    print(original_text)
    print(sorted(icd))
    print("\n".join(["%s: %s -> %s" % (i, sorted(icd2hpo[i]), sorted(icd2hpo_predcessor[i])) for i in icd if i in icd2hpo]))
    print(sorted(silver))
    print(sorted(keyword))
    print(evaluation.jaccard(set(silver), set(keyword)))

def analyze_num_children_node_hpo():
    children_info = dataloader.get_hpo_children_info()
    for idx, hpo_id in enumerate(dataloader.hpo_limited_list):
        print(idx, hpo_id, len(children_info[hpo_id]))

def analyze_ehr_pheno_ncbo_results():
    ncbo_data = baselines.ehr_phenolyzer_ncbo_annotator()

    ncbo = ncbo_data["HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY"].tolist()

    counter = 0
    for hpostr in ncbo:
        if isinstance(hpostr, float):
            continue
        if len(hpostr.split("/")) == 0:
            continue
        counter += 1
    print(counter, len(ncbo))

def analyze_threshold():

    import mimic_proc

    train_sentence2alpha = decision.load_sentence2alpha_mapping(corpus_to_analysis='train')
    # test_sentence2alpha = decision.load_sentence2alpha_mapping(corpus_to_analysis='test')
    test_sentence2alpha = {}

    sentence2alpha = {**train_sentence2alpha, **test_sentence2alpha}
    del train_sentence2alpha, test_sentence2alpha

    mimic_data = dataloader.load_mimic()
    mimic_text = mimic_data["CLEAN_TEXT"].tolist()

    ncbo_data = baselines.ehr_phenolyzer_ncbo_annotator()
    ncbo_results = ncbo_data["HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY"].tolist()

    limited_hpo_dict = dict()
    for idx, hpo_id in enumerate(dataloader.hpo_limited_list):
        limited_hpo_dict[hpo_id] = idx

    ncbo_results = [set([limited_hpo_dict[hpo_id] for hpo_id in hpostr.split("/")]) if isinstance(hpostr, str) else set() for hpostr in ncbo_results]


    hpo_alpha_positive = []
    hpo_alpha_negative = []
    for _ in dataloader.hpo_limited_list:
        hpo_alpha_negative.append([])
        hpo_alpha_positive.append([])

    for ehr_idx, ehr in enumerate(tqdm(mimic_text[:1000])):
        ehr_sentences = mimic_proc.get_sentences_from_mimic(ehr)

        alpha_results_list = list()

        for sent in ehr_sentences:
            if sent not in sentence2alpha:
                continue
            alpha_results_list.append(sentence2alpha[sent])
        if len(alpha_results_list) == 0:
            continue
        alpha_results_list = np.stack(alpha_results_list, axis=0)

        alpha_results_list = np.max(alpha_results_list, axis=0)
        assert alpha_results_list.shape[0] == len(dataloader.hpo_limited_list)

        for hidx in range(len(dataloader.hpo_limited_list)):
            if hidx in ncbo_results[ehr_idx]:
                hpo_alpha_positive[hidx].append(alpha_results_list[hidx])
            else:
                hpo_alpha_negative[hidx].append(alpha_results_list[hidx])

    for hidx in range(len(dataloader.hpo_limited_list)):
        print("===========")
        print(hidx, dataloader.hpo_limited_list[hidx])
        print("positive: %.3f %.3f %.3f" % (
            np.max(hpo_alpha_positive[hidx]),
            np.mean(hpo_alpha_positive[hidx]),
            np.min(hpo_alpha_positive[hidx])
        ))
        print("negative: %.3f %.3f %.3f" % (
            np.max(hpo_alpha_negative[hidx]),
            np.mean(hpo_alpha_negative[hidx]),
            np.min(hpo_alpha_negative[hidx])
        ))

def analyze_appearance_of_hpo(topk):

    children_info = dataloader.get_hpo_children_info()

    silver = evaluation.get_icd2hpo_3digit_results()
    '''
    threshold=[
                  0.3, 0.4, 0.2, 0.8,
                  0.4, 0.65, 0.4, 0.1,
                  0.85, 0.4, 0.2, 0.04,
                  0.01, 0.7, 0.4, 0.02,
                  0.7, 0.05, 0.4, 0.8,
                  0.2, 0.5, 0.6, 0.6
              ]
    '''
    threshold=[
        0.3, 0.35, 0.08, 0.7,
        0.4, 0.7, 0.25, 0.1,
        0.85, 0.4, 0.2, 0.03,
        0.01, 0.7, 0.4, 0.02,
        0.85, 0.05, 0.4, 0.8,
        0.15, 0.55, 0.65, 0.6
    ]
    '''
    threshold = [
        0.308, 0.378, 0.229, 0.968,
        0.727, 0.595, 0.288, 0.031,
        0.577, 0.713, 0.315, 0.056,
        0.002, 0.531, 0.689, 0.003,
        0.557, 0.059, 0.479, 0.806,
        0.093, 0.571, 0.685, 0.554
    ]
    '''

    column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"

    decision_mode = 'var_threshold'
    # threshold = analyze_alpha()
    unsuper = decision.results_of_alpha_out(threshold=threshold, mode=decision_mode)[column_of_keyword].tolist()
    return unsuper

    # unsuper = decision.results_of_alpha_out_norm_topk(topk)[column_of_keyword].tolist()

    if config._global_verbose_print:
        utils.print_statictis_of_hpo(silver)
        utils.print_statictis_of_hpo(unsuper)

    '''
    evaluation._evaluate(
        [silver[index] for index in config.mimic_train_indices],
        [unsuper[index] for index in config.mimic_train_indices],
        func=evaluation.jaccard
    )
    evaluation._evaluate(
        [silver[index] for index in config.mimic_train_indices],
        [unsuper[index] for index in config.mimic_train_indices],
        func=evaluation.overlap_coefficient
    )
    '''

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

    counter_silver = [0] * len(dataloader.hpo_limited_list)
    counter_unsuper = [0] * len(dataloader.hpo_limited_list)

    hpo2hpoidx = {hpo: hidx for hidx, hpo in enumerate(dataloader.hpo_limited_list)}

    assert len(silver) == len(unsuper)
    for i in range(len(silver)):
        if not isinstance(silver[i], str) or not isinstance(unsuper[i], str):
            continue
        silver_hpolist = [s for s in silver[i].split("/") if len(s) > 0]
        unsuper_hpolist = [s for s in unsuper[i].split("/") if len(s) > 0]

        for hpo in silver_hpolist:
            counter_silver[hpo2hpoidx[hpo]] += 1
        for hpo in unsuper_hpolist:
            counter_unsuper[hpo2hpoidx[hpo]] += 1

    for hidx in range(len(dataloader.hpo_limited_list)):
        if config._global_verbose_print:
            print(hidx, dataloader.hpo_limited_list[hidx], counter_silver[hidx], counter_unsuper[hidx], len(children_info[dataloader.hpo_limited_list[hidx]]))

def analyze_alpha():

    with open(config.outputs_results_dir + "mimic_alpha_%s_55000.npy" % 'train', 'rb') as f:
        mimic_alpha_train_results = np.load(f)

    new_threshold = []
    for i in range(len(dataloader.hpo_limited_list)):
        hpo_alpha = utils.sigmoid(mimic_alpha_train_results[:, i])
        mean = np.mean(hpo_alpha)
        std = np.std(hpo_alpha)
        normalized_hpo_alpha = (hpo_alpha - mean) / std
        # print((threshold[i] - mean) / std)
        # print(np.mean(normalized_hpo_alpha))
        # print(np.std(normalized_hpo_alpha))
        nt = 1.28 * std + mean
        # nt = 1.04 * std + mean
        # nt = 0.84 * std + mean
        # nt = 0.53 * std + mean
        new_threshold.append(nt)
        # print("----")

    if config._global_verbose_print:
        print(", ".join(["%.3f" % nt for nt in new_threshold]) )
    return new_threshold

def analyze_alpha_2():

    with open(config.outputs_results_dir + "mimic_alpha_%s_55000.npy" % 'train', 'rb') as f:
        mimic_alpha_train_results = np.load(f)

    threshold=[
        0.3, 0.35, 0.08, 0.7,
        0.4, 0.7, 0.25, 0.1,
        0.85, 0.4, 0.2, 0.03,
        0.01, 0.7, 0.4, 0.02,
        0.85, 0.05, 0.4, 0.8,
        0.15, 0.55, 0.65, 0.6
    ]

    silver = evaluation.get_icd2hpo_3digit_results()
    counter_silver = [0] * len(dataloader.hpo_limited_list)

    hpo2hpoidx = {hpo: hidx for hidx, hpo in enumerate(dataloader.hpo_limited_list)}

    for i in range(len(silver)):
        if not isinstance(silver[i], str):
            continue
        silver_hpolist = [s for s in silver[i].split("/") if len(s) > 0]

        for hpo in silver_hpolist:
            counter_silver[hpo2hpoidx[hpo]] += 1

    for i in range(len(dataloader.hpo_limited_list)):
        hpo_alpha = utils.sigmoid(mimic_alpha_train_results[:, i])
        mean = np.mean(hpo_alpha)
        std = np.std(hpo_alpha)
        normalized_hpo_alpha = (hpo_alpha - mean) / std

        position = (threshold[i] - mean) / std
        percent = counter_silver[i] / config.total_num_mimic_record
        p = "%2d %.3f %.3f %.3f %.3f" % (i, threshold[i], position, percent, percent / position)
        print(p)

def mapping_hpo_annotation_to_icd():

    icd2limitedhpo_mapping = dataloader.get_3digit_icd_hpo_in_limited_hpo_set()

    hpo2icd_policy1 = dict()
    for icd in icd2limitedhpo_mapping:
        for hpo in icd2limitedhpo_mapping[icd]:
            if hpo not in hpo2icd_policy1:
                hpo2icd_policy1[hpo] = set()
            hpo2icd_policy1[hpo].add(icd)

    df = pd.read_csv(config.outputs_results_dir + "unsupervised_method_var_threshold_0.3975.csv")

    icd2ehr = dict()

    def update_icd2ehr(icd9_set, column):
        for icd in icd9_set:
            if icd not in icd2ehr:
                icd2ehr[icd] = {
                    "original": list(),
                    "policy1": list(),
                    "policy2": list()
                }
            icd2ehr[icd][column].append(index)

    for index, row in df.iterrows():
        icd9_codes = row["ICD9_CODE_LIST"]
        hpo_codes = row["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"]

        icd9_codes = set() if type(icd9_codes) != str else set([icd[:3] for icd in icd9_codes.split("/")])
        hpo_codes = set() if type(hpo_codes) != str else set([hpo for hpo in hpo_codes.split("/")])

        update_icd2ehr(icd9_codes, "original")

        icd9_codes_with_policy1 = set([icd for hpo in hpo_codes for icd in hpo2icd_policy1[hpo]])
        update_icd2ehr(icd9_codes_with_policy1, "policy1")

        icd9_codes_with_policy2 = set([icd for icd in icd2limitedhpo_mapping if icd2limitedhpo_mapping[icd] <= hpo_codes])
        update_icd2ehr(icd9_codes_with_policy2, "policy2")

    with open("HPO2ICD_POLICY12.csv", 'w') as outfile:
        outfile.write("%s\t%s\t%s\t%s\n" % ("ICD", "Original", "Policy1", "Policy2"))
        for icd in icd2limitedhpo_mapping:
            outfile.write("%s\t%d\t%d\t%d\n" % (icd, len(icd2ehr[icd]["original"]), len(icd2ehr[icd]["policy1"]), len(icd2ehr[icd]["policy2"])))

    print(icd2limitedhpo_mapping['277'])
    print(icd2limitedhpo_mapping['153'])
    print(icd2limitedhpo_mapping['287'])

def analyze_avg_icd_per_ehr():
    df = pd.read_csv(baselines.silver_standard_results_file)
    icd_series = df["ICD9_CODE_LIST"]
    icd2hpo = dataloader.get_3digit_icd_hpo_in_limited_hpo_set()

    num_icd_list = []

    for idx, value in icd_series.iteritems():
        if type(value) is not str:
            continue
        icdlist = set([icd[:3] for icd in value.split("/") if icd[:3] in icd2hpo])
        num_icd_list.append(len(icdlist))

    print(np.mean(num_icd_list))
    print(np.median(num_icd_list))

if __name__ == '__main__':
    # icd_distribution_in_mimic()
    # num_mimic_negation()
    # avg_hpo_per_sentence()
    # hpo_have_most_sentences()
    # analysis_sentence_embedding()
    # analysis_hpo_embedding(level=1)
    # analysis_hpo_embedding(level=2)
    # analysis_hpo_embedding(level=1, root_node="HP:0000118")
    # analysis_alpha_out(corpus_to_analysis='train')
    # analysis_alpha_out(corpus_to_analysis='train')
    # analysis_alpha_out(corpus_to_analysis='test')
    # description_of_hpo()

    # avg_hpo_per_sentence2()

    '''
    distribution_of_hpo_in_silver_standard(baselines.silver_standard()['HPO_CODE_LIST'].tolist())
    print("----")
    distribution_of_hpo_in_silver_standard(decision.results_of_alpha_out(threshold=0.5, mode='argmax')["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"].tolist())
    print("----")
    distribution_of_hpo_in_silver_standard(decision.results_of_alpha_out(threshold=0.1, mode='doc')["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"].tolist())
    '''

    # analysis_overlapping_of_children_node()
    # hpo_description_optimization()
    # convert_mimic_to_plain_text()
    # analyze_ehr_phenolyzer_results()
    # analyze_specific_case(385)
    # hpodata = dataloader.get_hpo4dataset()
    # print(hpodata['HP:0000408']['terms'])
    # analyze_num_children_node_hpo()
    # analyze_ehr_pheno_ncbo_results()

    # analyze_threshold()
    # exit()

    # ncbo = baselines.ehr_phenolyzer_ncbo_annotator()
    # obo = baselines.obo_annotator()

    # ncbo = ncbo[[
    #     "ICD9_CODE_LIST",
    #     "HPO_CODE_LIST_EHR_PHENO",
    #     "HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY",
    # ]]
    # obo = obo[[
    #     "ICD9_CODE_LIST",
    #     "HPO_CODE_LIST_OBO_ANNO",
    #     "HPO_CODE_LIST_OBO_ANNO_PREDECESSORS_ONLY",
    # ]]

    # ncbo.to_csv("ncbo.csv")
    # obo.to_csv("obo.csv")

    # candidate_icd = ['401', '272', '276', '285', '410']
    # icd2limitedhpo_mapping = dataloader.get_3digit_icd_hpo_in_limited_hpo_set()
    # for icd in candidate_icd:
    #     print(icd, sorted(icd2limitedhpo_mapping[icd]))

    '''
    children_info = dataloader.get_hpo_children_info()
    hposet = set()
    for hpo in dataloader.hpo_limited_list:
        # hposet.add(hpo)
        hposet.update(children_info[hpo])
    print(len(hposet))
    exit()
    '''

    # analyze_appearance_of_hpo(8)

    # analyze_alpha()
    # analyze_alpha_2()

    '''
    silver = evaluation.get_icd2hpo_3digit_results()
    counter = 0
    for hpostr in silver:
        if not isinstance(hpostr, str) or len(hpostr) == 0:
            continue
        counter += 1
    print(counter, len(silver))
    '''

    '''
    icd2limitedhpo_mapping = dataloader.get_3digit_icd_hpo_in_limited_hpo_set()
    print(len(icd2limitedhpo_mapping))
    counter = 0
    for icd in icd2limitedhpo_mapping:
        counter += len(icd2limitedhpo_mapping[icd])
    print(counter)
    '''

    '''
    silver = evaluation.get_icd2hpo_3digit_results()
    candidate = [42292, 19235, 22338]
    import pandas as pd
    df = pd.read_csv(config.outputs_results_dir + "unsupervised_method_var_threshold_0.3975.csv")
    # hpostr = df["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"][]
    df = df.iloc[candidate]
    df["HPO_ANNOTATION"] = df["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"]
    df["EHR_TEXT"] = df["TEXT"]
    df = df[["HPO_ANNOTATION", "ICD9_CODE_LIST", "EHR_TEXT"]]
    df.to_csv("samples.csv")
    print(df)
    '''

    # mapping_hpo_annotation_to_icd()
    analyze_avg_icd_per_ehr()

    pass
