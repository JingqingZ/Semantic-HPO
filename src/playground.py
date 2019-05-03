## The code in this file is used for analysis (or playground)
import pickle
from tqdm import tqdm
import numpy as np

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
    hpo_description_optimization()
    pass
