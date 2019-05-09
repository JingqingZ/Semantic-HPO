# Dataloader
# Data loading and early analysis on data

import os
import re
import json
import pickle
import pronto
import pandas as pd
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

import config

# environment variables, subject to changes according to local environment
# the folder that contains original data
data_dir = "../data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# the path to HPO ontology file
# download link: https://hpo.jax.org/app/download/ontology
hpo_ontology_file = data_dir + "HPO/hp.owl"
hpo_omim_file = data_dir + "HPO/phenotype_annotation_hpoteam.tab"
# the path to MIMIC-III files
# download link: https://mimic.physionet.org/
# application to access the data required
mimic_note_file = data_dir + "MIMIC/NOTEEVENTS.csv"
mimic_diag_file = data_dir + "MIMIC/DIAGNOSES_ICD.csv"
mimic_icd_file = data_dir + "MIMIC/D_ICD_DIAGNOSES.csv"
# the path to the OrphaData
# download link: http://www.orphadata.org/cgi-bin/index.php
# phenotypes associated with rare disorders
orpha_hpo_file = data_dir + "orphadata/en_product4_HPO.xml"
# the path to other dataset
# supplementary files of paper: Goh, Kwang-Il, et al. "The human disease network." Proceedings of the National Academy of Sciences 104.21 (2007): 8685-8690.
# supplementary files of paper: Park, Juyong, et al. "The impact of cellular networks on disease comorbidity." Molecular systems biology 5.1 (2009): 262.
# The files need to be manually converted from xslx / pdf to csv
goh2omim_file = data_dir + "others/goh2omim.csv"
goh2icd_file = data_dir + "others/goh2icd.csv"

# intermediate files which will be created
# created by load_hpo_description()
hpo_description_file = data_dir + "HPO/description.json"
# create by get_hpo4dataset()
# this file connects mimic and hpo via keywords search
hpo_dataset_file = data_dir + "HPO/dataset.pickle"
hpo_children_info_file = data_dir + "HPO/children.pickle"
# OMIM to HPO mapping
omim2hpo_file = data_dir + "HPO/omim2hpo.json"
# ICD to OMIM mapping
icd2omim_file = data_dir + "HPO/icd2omim.json"
# created by load_mimic()
mimic_file = data_dir + "MIMIC/ready.csv"
# created get_corpus()
text_corpus_mimic_file = data_dir + "corpus_mimic.txt"
text_corpus_hpo_file = data_dir + "corpus_hpo.json"
# created get_vocab()
vocab_file = data_dir + "vocab.txt"
vocab_for_analysis_file = data_dir + "vocab4analysis.csv"
# the mapping from rare disease to hpo
# deprecated
disease2hpo_map_file = data_dir + "disease2hpo.csv"
icd2hpo_map_file = data_dir + "icd2hpo.csv"

hpo_domains = [
    'name',
    'desc',
    'other/comment',
    'other/hasExactSynonym',
    'other/hasRelatedSynonym'
]

hpo_root_id = "HP:0000001"
hpo_phenotypic_abnormality_id = "HP:0000118"
hpo_limited_list = [
    'HP:0000707', 'HP:0000478', 'HP:0001507', 'HP:0001626',
    'HP:0001939', 'HP:0025354', 'HP:0025031', 'HP:0000598',
    'HP:0003549', 'HP:0001574', 'HP:0000152', 'HP:0000924',
    'HP:0001608', 'HP:0002664', 'HP:0001871', 'HP:0000769',
    'HP:0025142', 'HP:0000818', 'HP:0000119', 'HP:0001197',
    'HP:0040064', 'HP:0003011', 'HP:0002715', 'HP:0002086'
]

def load_hpo_ontology():
    print("Loading HPO ontology ...")
    ont = pronto.Ontology(hpo_ontology_file)
    data = json.loads(ont.json)
    return data

def load_hpo_raw_description():
    import queue

    if not os.path.exists(hpo_description_file):
    # if True:

        print("Loading HPO description ...")

        data = load_hpo_ontology()
        process_data = dict()

        node_queue = queue.Queue()
        node_queue.put(hpo_root_id)

        while not node_queue.empty():
            node = node_queue.get()
            process_data[node] = dict()
            for d in hpo_domains:
                if "/" in d:
                    d1 = d.split("/")[0]
                    d2 = d.split("/")[1]
                    text = ", ".join(data[node].get(d1, {}).get(d2, []))
                else:
                    text = data[node].get(d, "")

                process_data[node][d] = text

            children = data[node]["relations"].get("can_be", [])
            for cnode in children:
                node_queue.put(cnode)

        with open(hpo_description_file, 'w') as outfile:
            json.dump(process_data, outfile)

        print("HPO descriptions have been extracted and save to %s" % hpo_description_file)

    else:

        print("Loading HPO descriptions from file %s" % hpo_description_file)

        with open(hpo_description_file, 'r') as infile:
            data = json.load(infile)

    return data

# deprecated
def convert_hpo_to_docs():
    print("Converting HPO descriptions to documents ...")

    ontology = load_hpo_ontology()
    description = load_hpo_raw_description()

    docs = list()

    def dfs(node, text):
        desc_of_node = " ".join([description[node][d] for d in hpo_domains])
        if text == "":
            newtext = desc_of_node
        else:
            newtext = text + "\n" + desc_of_node

        children = ontology[node]["relations"].get("can_be", [])
        if len(children) == 0:
            docs.append(newtext.split("\n"))
        else:
            for child in children:
                dfs(child, newtext)

    dfs(hpo_root_id, "")

    return docs

def convert_hpo_to_flat_description():

    description = load_hpo_raw_description()

    new_desc_dict = dict()

    for node in description:
        desc_of_node = " ".join([description[node][d] for d in hpo_domains])
        new_desc_dict[node] = desc_of_node

    return new_desc_dict

def plot_hpo_ontology():

    data = load_hpo_ontology()

    from pyvis.network import Network
    import queue
    net = Network()

    # HP:0000001 All
    # HP:0000018 Phenotypic Abnormality
    # Other branch is also necessary

    node_queue = queue.Queue()
    node_queue.put(hpo_root_id)
    net.add_node(hpo_root_id)

    node_used = 1

    while not node_queue.empty():
        node = node_queue.get()
        children = data[node]["relations"].get("can_be", [])
        for cnode in children:
            node_queue.put(cnode)
            net.add_node(cnode)
            net.add_edge(node, cnode)
            node_used += 1
        if node_used > 300:
            break

    net.show("hpo.html")

def load_mimic():

    if not os.path.exists(mimic_file):

        def _merge_note(df):
            text = ""
            for idx, row in df.iterrows():
                text += row["TEXT"]
            ndf = pd.DataFrame(
                data={
                    "HADM_ID":[df["HADM_ID"].iloc[0]],
                    "SUBJECT_ID": [df["SUBJECT_ID"].iloc[0]],
                    "TEXT": text
                }
            )
            return ndf

        print("Loading and processing MIMIC note data ...")

        note_data = pd.read_csv(mimic_note_file)
        note_data = note_data[note_data["CATEGORY"].str.contains("Discharge summary")]
        note_data = note_data[["SUBJECT_ID", "HADM_ID", "TEXT"]]
        note_data["SUBJECT_ID"] = note_data["SUBJECT_ID"].astype(str, copy=False)
        note_data["HADM_ID"] = note_data["HADM_ID"].astype(str, copy=False)
        note_data = note_data.groupby("HADM_ID").apply(_merge_note).reset_index(drop=True)

        assert not any(note_data["HADM_ID"].duplicated())
        note_data = note_data.set_index("HADM_ID")

        note_data.to_csv(mimic_note_file + "_processed.csv")

        def _merge_icd(df):
            max_seq = df["SEQ_NUM"].iloc[-1]
            for idx, row in df.iterrows():
                max_seq = max(row["SEQ_NUM"], max_seq)
            icd_code_list = [None] * max_seq
            for idx, row in df.iterrows():
                icd_code_list[row["SEQ_NUM"] - 1] = row["ICD9_CODE"]
            icd_code_list = [icd for icd in icd_code_list if icd is not None]
            ndf = pd.DataFrame(
                data={
                    "HADM_ID": [df["HADM_ID"].iloc[0]],
                    "ICD9_CODE_LIST": ["/".join(icd_code_list)]
                },
            )
            return ndf

        print("Loading and processing MIMIC ICD diagnosis data ...")

        diag_data = pd.read_csv(mimic_diag_file)
        diag_data = diag_data[["HADM_ID", "SEQ_NUM", "ICD9_CODE"]]
        diag_data = diag_data.dropna()
        diag_data["HADM_ID"] = diag_data["HADM_ID"].astype(str, copy=False)
        diag_data["ICD9_CODE"] = diag_data["ICD9_CODE"].astype(str, copy=False)
        diag_data["SEQ_NUM"] = diag_data["SEQ_NUM"].astype(int, copy=False)
        diag_data = diag_data.groupby("HADM_ID").apply(_merge_icd).reset_index(drop=True)
        diag_data = diag_data.set_index("HADM_ID")
        # print(diag_data)

        diag_data.to_csv(mimic_diag_file + "_processed.csv")

        # the extreme size of diag_data and note_data may lead to failure of this step
        # If failed, load processed diag_data and note_data from corresponding xxx_processed.csv files
        # then merge them and save.
        # sound tricky to be honest.
        result = pd.merge(diag_data, note_data, on='HADM_ID')
        # print(result)

        result.to_csv(mimic_file)
        print("MIMIC note data and ICD diagnosis data have been merged and saved to %s" % mimic_file)

    else:
        print("Loading MIMIC note data and ICD diagnosis data from %s" % mimic_file)
        result = pd.read_csv(mimic_file)

    return result

def get_corpus(most_common=config.vocabulary_size, sentence_length=config.sequence_length // 2, rebuild=False):

    from collections import Counter
    from tqdm import tqdm
    counter = Counter()
    counter_mimic = Counter()
    counter_hpo = Counter()

    if not os.path.exists(vocab_file) or not os.path.exists(text_corpus_mimic_file) or \
            not os.path.exists(text_corpus_hpo_file) or rebuild:
        # white_space = [" ", "\n", "\r", "\t", "\f", "\v"]
        # special_token_to_ignore = ["?"]

        mimic_data = load_mimic()
        text_data = mimic_data["TEXT"].tolist()
        # TODO: load ICD code data

        mimic_doc_list = []
        print("Processing the original MIMIC data and creating vocabulary ...")
        for doc in tqdm(text_data[:]):
            # processing the text
            # doc = doc.strip()
            doc = doc.lower()
            regex = re.compile('[^a-z\s]')
            doc = regex.sub(" ", doc)
            # doc = re.sub(r"\[\*\*.*\*\*\]", "", doc)
            # doc = nlp(doc)
            tlist = []
            doc = doc.split()
            for token in doc:
                text = token.strip()
                # count the appearance of words
                if text != "":
                    counter[text] += 1
                    counter_mimic[text] += 1
                tlist.append(text)
            mimic_doc_list.append(tlist)

        # hpo_data = convert_hpo_to_docs()
        hpo_data = convert_hpo_to_flat_description()
        hpo_doc_dict = dict()
        print("Processing the HPO data and creating vocabulary ...")
        for node in tqdm(hpo_data):
            sentence = hpo_data[node]
            sentence = sentence.lower()
            regex = re.compile('[^a-z\s]')
            sentence = regex.sub(" ", sentence)

            tokens = sentence.split()
            for token in tokens:
                text = token.strip()
                if text != "":
                    counter[text] += 1
                    counter_hpo[text] += 1

            hpo_doc_dict[node] = sentence

        # get most frequent words
        # all words in hpo will be included in vocab
        vocab_hpo = [x for x in counter_hpo.items() if x[1] >= 1]
        word_counts = dict([(w[0], counter[w[0]]) for w in vocab_hpo])

        all_counts = [x for x in counter.items() if x[1] >= 1]
        all_counts.sort(key=lambda x: x[1], reverse=True)
        all_counts = all_counts[:most_common]

        for word_tuple in all_counts:
            if len(word_counts) >= most_common:
                break
            word = word_tuple[0]
            if word not in word_counts:
                word_counts[word] = word_tuple[1]

        vocab_sorted = sorted(word_counts.items(), key=lambda w: w[1], reverse=True)

        # saving
        with open(vocab_file, 'w') as out_vocab_file:
            for word in config.special_tokens_in_vocab:
                out_vocab_file.write(word + "\n")
            for word in vocab_sorted:
                out_vocab_file.write(word[0] + "\n")
        print("Vocab saved to %s" % vocab_file)

        with open(vocab_for_analysis_file, 'w') as out_vocab_file:
            out_vocab_file.write("word,mimic,hpo,total\n")
            for word in config.special_tokens_in_vocab:
                out_vocab_file.write(word + ",,,\n")
            for word in vocab_sorted:
                out_vocab_file.write(word[0] + ",%d,%d,%d" % (counter_mimic[word[0]], counter_hpo[word[0]], counter[word[0]]) + "\n")
        print("Vocab (for analysis) saved to %s" % vocab_for_analysis_file)

        # analysis
        vocab_include_mimic = 0
        vocab_notinclude_mimic = 0
        for word, appearance in counter_mimic.items():
            if word in word_counts:
                vocab_include_mimic += 1
            else:
                vocab_notinclude_mimic += 1
        print("Summary: %d (%d) words in MIMIC are (not) included in vocab." % (vocab_include_mimic, vocab_notinclude_mimic))

        # analysis
        vocab_include_hpo = 0
        vocab_notinclude_hpo = 0
        for word, appearance in counter_hpo.items():
            if word in word_counts:
                vocab_include_hpo += 1
            else:
                vocab_notinclude_hpo += 1
        print("Summary: %d (%d) words in HPO are (not) included in vocab." % (vocab_include_hpo, vocab_notinclude_hpo))

        # based the vocabulary just got
        # cleaning the original textual data
        # the words that are not included in vocab will be masked as [UNK]
        print("Cleaning the original MIMIC data ...")
        clean_mimic_doc_list = []
        for doc in tqdm(mimic_doc_list[:]):
            token_list = []
            for token in doc:
                text = token.strip()
                if text == "":
                    pass
                elif text in word_counts:
                    token_list.append(text)
                else:
                    token_list.append("[UNK]")
            new_doc = ""
            for tidx, token in enumerate(token_list):
                new_doc += token
                if (tidx + 1) % sentence_length == 0:
                    new_doc += "\n"
                elif tidx < len(token_list) - 1:
                    new_doc += " "
            if not new_doc.endswith("\n"):
                new_doc += "\n"

            clean_mimic_doc_list.append(new_doc)

        print("Cleaning the HPO data ...")
        clean_hpo_doc_dict = dict()
        for node in tqdm(hpo_doc_dict):
            sentence = hpo_doc_dict[node]

            token_list = []
            tokens = sentence.split()
            for token in tokens:
                text = token.strip()
                if text == "":
                    pass
                elif text in word_counts:
                    token_list.append(text)
                else:
                    token_list.append("[UNK]")
            new_sentence = " ".join(token_list)

            clean_hpo_doc_dict[node] = new_sentence

        with open(text_corpus_mimic_file, 'w') as out_corpus_file:
            for doc in clean_mimic_doc_list:
                out_corpus_file.write(doc + "==DOCSPLIT==\n")
        print("Corpus of MIMIC saved to %s" % text_corpus_mimic_file)

        with open(text_corpus_hpo_file, 'w') as out_corpus_file:
            json.dump(clean_hpo_doc_dict, out_corpus_file)
        print("Corpus of HPO saved to %s" % text_corpus_hpo_file)

        assert mimic_data.shape[0] == len(mimic_doc_list)
        mimic_data['CLEAN_TEXT'] = clean_mimic_doc_list
        mimic_data.to_csv(mimic_file)
        print("Update %s with cleaned text" % mimic_file)

    else:
        with open(text_corpus_mimic_file, 'r') as infile:
            clean_mimic_doc_list = infile.read()
        print("Corpus of MIMIC loaded from %s" % text_corpus_mimic_file)
        clean_mimic_doc_list = clean_mimic_doc_list.split("==DOCSPLIT==\n")
        if len(clean_mimic_doc_list[-1].strip()) == 0:
            clean_mimic_doc_list = clean_mimic_doc_list[:-1]

        with open(text_corpus_hpo_file, 'r') as infile:
            clean_hpo_doc_dict = json.load(infile)
        print("Corpus of HPO loaded from %s" % text_corpus_hpo_file)

        total_num_of_sentence = 0
        for doc in clean_mimic_doc_list:
            total_num_of_sentence += len(doc.split("\n"))

        print("Total num of MIMIC docs: %d" % len(clean_mimic_doc_list))
        print("Total num of MIMIC sentences: %d" % total_num_of_sentence)
        # Total num of MIMIC docs: 52722
        # Total num of MIMIC sentences: 2575124

    hpo_onto = load_hpo_ontology()

    for hpo in hpo_limited_list:
        children = hpo_onto[hpo]["relations"].get("can_be", [])
        if len(clean_hpo_doc_dict[hpo].split()) < config.sequence_length // 2:
            origin_word_set = set(["abnormality", "of"])
            newwordlist = list()
            for cnode in children:
                for word in clean_hpo_doc_dict[cnode].split():
                    if word not in origin_word_set:
                        newwordlist.append(word)
            new_desc = "%s %s" % (
                clean_hpo_doc_dict[hpo],
                " ".join(newwordlist)
            )
        else:
            new_desc = clean_hpo_doc_dict[hpo]
        clean_hpo_doc_dict[hpo] =new_desc
    del hpo_onto

    return clean_mimic_doc_list, clean_hpo_doc_dict

def get_raredisease2hpo_mapping():

    if not os.path.exists(disease2hpo_map_file):

        print("Processing disease to hpo mapping ... ")

        tree = ET.parse(orpha_hpo_file)
        root = tree.getroot()

        mapping = list()

        disorderlist = root[1]
        for disease in tqdm(disorderlist):
            disease_name = disease.find('Name').text
            disease_id = disease.attrib['id']
            for hpo in disease.find('HPODisorderAssociationList'):
                hpo_id = hpo.find('HPO').find('HPOId').text
                hpo_term = hpo.find('HPO').find('HPOTerm').text
                hpo_frequency = hpo.find('HPOFrequency').find('Name').text
                mapping.append([disease_name, disease_id, hpo_id, hpo_term, hpo_frequency])

        df = pd.DataFrame(data=mapping, columns=["disease_name", 'disease_id', 'hpo_id', 'hpo_term', 'hpo_frequency'])
        df.to_csv(disease2hpo_map_file)
        print("Saving disease to hpo mapping to %s" % disease2hpo_map_file)

    else:

        print("Loading disease to hpo mapping from %s" % disease2hpo_map_file)
        df = pd.read_csv(disease2hpo_map_file)

    return df

'''
def get_icd2hpo_mapping():

    disease2hpo_df = get_disease2hpo_mapping()
    icd_df = pd.read_csv(mimic_icd_file)

    for rowidx, row in disease2hpo_df.iterrows():
        print(row)
        exit()
'''

def get_hpo4dataset():

    if not os.path.exists(hpo_dataset_file):
        print("Note: This function may take a long time to process. So be patient.")

        hpo_ontology = load_hpo_ontology()
        hpo_raw_description = load_hpo_raw_description()

        corpus_mimic_data, hpo_description = get_corpus()
        # train_corpus_mimic = corpus_mimic_data[:int(len(corpus_mimic_data) * config.training_percentage)]
        # test_corpus_mimic = corpus_mimic_data[-int(len(corpus_mimic_data) * config.testing_percentage):]
        assert len(corpus_mimic_data) == config.total_num_mimic_record
        train_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_train_indices]
        test_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_test_indices]

        hpo_dataset = dict()
        # hpo_dataset is dict
        # hpo_dataset[node] is dict, where node represent a HPO term
        # hpo_dataset[node]['status'] True or False, some nodes will be deactivated to use for modelling
        # hpo_dataset[node]['depth'] the depth of the current node
        # hpo_dataset[node]['description'] is the description of this HPO term
        # hpo_dataset[node]['terms'] is a list of terms closely related with the node
        # hpo_dataset[node]['children_node'] is a list of all children term of the node
        # hpo_dataset[node][['mimic_train'] is a list of str from mimic, the name / synonym of the HPO term is mentioned in the str (keyword search)
        # hpo_dataset[node][['mimic_test'] is a list of str from mimic, the name / synonym of the HPO term is mentioned in the str (keyword search)

        hpo_terms_bank = dict()

        print("Processing hpo data and create a dataset...")

        with tqdm(total=len(hpo_description)) as pbar:

            def dfs(node, depth):
                pbar.update(len(hpo_dataset))

                # desc_of_node = " ".join([hpo_description[node][d] for d in hpo_domains])
                # desc_of_node = desc_of_node.strip().lower()
                # regex = re.compile('[^a-z\s]')
                # desc_of_node = regex.sub(" ", desc_of_node)
                desc_of_node = hpo_description[node]

                # assert hpo is a tree
                if node in hpo_dataset:
                    return
                # actually hpo is not a tree

                hpo_dataset[node] = dict()
                hpo_dataset[node]['status'] = True if depth >= 1 else False
                hpo_dataset[node]['depth'] = depth
                hpo_dataset[node]["description"] = desc_of_node
                hpo_dataset[node]["children_node"] = set()

                related_terms = set()
                related_terms.add(hpo_raw_description[node]["name"].lower())
                related_terms.update([t for t in [term.strip().lower() for term in hpo_ontology[node].get('other', {}).get('hasExactSynonym', [])] if len(t) != 0]) # not empty str
                related_terms.update([t for t in [term.strip().lower() for term in hpo_ontology[node].get('other', {}).get('hasRelatedSynonym', [])] if len(t) != 0]) # no emtpy str
                hpo_dataset[node]["terms"] = related_terms

                hpo_dataset[node]['mimic_train'] = set()
                hpo_dataset[node]['mimic_test'] = set()

                children = hpo_ontology[node]["relations"].get("can_be", [])
                for child in children:
                    hpo_dataset[node]["children_node"].add(child)
                    dfs(child, depth + 1)
                    hpo_dataset[node]["children_node"].update(hpo_dataset[child]["children_node"])

            dfs(hpo_root_id, 0)

        def get_sentence(term, data):
            sentence_set = set()
            for doc in data:
                sentences = doc.split("\n")
                for sentence in sentences:
                    if term in sentence:
                        sentence_set.add(sentence)
            return sentence_set

        for node in hpo_dataset:
            if hpo_dataset[node]['status']:
                for term in hpo_dataset[node]['terms']:
                    if term not in hpo_terms_bank:
                        hpo_terms_bank[term] = []
                    hpo_terms_bank[term].append(node)
                    assert len(hpo_terms_bank[term]) == 1  # tricky, each term corresponds with exact one hpo node

        for term in tqdm(hpo_terms_bank):
            for node in hpo_terms_bank[term]:
                hpo_dataset[node]['mimic_train'].update(get_sentence(term, train_corpus_mimic))
                hpo_dataset[node]['mimic_test'].update(get_sentence(term, test_corpus_mimic))


        with open(hpo_dataset_file, 'wb') as f:
            pickle.dump(hpo_dataset, f)

        print("Saving HPO dataset to %s" % hpo_dataset_file)

    else:

        print("Loading HPO dataset from %s" % hpo_dataset_file)

        with open(hpo_dataset_file, 'rb') as f:
            hpo_dataset = pickle.load(f)


    return hpo_dataset

def get_hpo_children_info():

    if not os.path.exists(hpo_children_info_file):
        hpo_children_info = dict()
        hpo_data = get_hpo4dataset()
        for hpo in hpo_data:
            hpo_children_info[hpo] = hpo_data[hpo]["children_node"]

        with open(hpo_children_info_file, 'wb') as f:
            pickle.dump(hpo_children_info, f)

    else:

        with open(hpo_children_info_file, 'rb') as f:
            hpo_children_info = pickle.load(f)
    return hpo_children_info

def get_omim_hpo_mapping():

    if not os.path.exists(omim2hpo_file):

        print("Processing the mapping between OMIM and HPO ...")

        omim_hpo_mapping = dict()

        with open(hpo_omim_file, 'r') as infile:
            data = infile.read()
            lines = data.split("\n")
            for line in lines:
                fields = line.split("\t")
                if fields[0] != 'OMIM':
                    continue
                omim_id = "%s:%s" % (fields[0], fields[1])
                hpo_id = fields[4]
                if omim_id not in omim_hpo_mapping:
                    omim_hpo_mapping[omim_id] = list()
                omim_hpo_mapping[omim_id].append(hpo_id)

        print("Saving OMIM to HPO mapping to %s" % omim2hpo_file)
        with open(omim2hpo_file, 'w') as f:
            json.dump(omim_hpo_mapping, f)
    else:

        print('Loading OMIM to HPO mapping from %s' % omim2hpo_file)
        with open(omim2hpo_file, 'r') as f:
            omim_hpo_mapping = json.load(f)

    return omim_hpo_mapping

def get_icd_omim_mapping():


    if not os.path.exists(icd2omim_file):

        print("Processing ICD to OMIM mapping...")

        icd_omim_mapping = dict()
        icd_goh = dict()
        goh_omim = dict()

        goh_icd_df = pd.read_csv(goh2icd_file, dtype=str)
        for idx, row in goh_icd_df.iterrows():
            goh_id = row[0]
            icd_id = row[2].replace(".", "").replace("[", "").replace("]", "")
            if icd_id not in icd_goh:
                icd_goh[icd_id] = set()
            icd_goh[icd_id].add(goh_id)

        goh_omim_df = pd.read_csv(goh2omim_file, dtype=str)
        for idx, row in goh_omim_df.iterrows():
            goh_id = row[0]
            omim_id = "OMIM:%s" % (row[3])
            if goh_id not in goh_omim:
                goh_omim[goh_id] = set()
            goh_omim[goh_id].add(omim_id)

        for icd_id in icd_goh:
            icd_omim_mapping[icd_id] = list()
            for goh_id in icd_goh[icd_id]:
                if goh_id not in goh_omim:
                    continue
                omim_id_set = goh_omim[goh_id]
                icd_omim_mapping[icd_id] += list(omim_id_set)
            if len(icd_omim_mapping[icd_id]) == 0:
                del icd_omim_mapping[icd_id]

        print("Saving ICD to OMIM mapping to %s" % icd2omim_file)
        with open(icd2omim_file, 'w') as f:
            json.dump(icd_omim_mapping, f)

    else:
        print("Loading ICD to OMIM mapping from %s" % icd2omim_file)
        with open(icd2omim_file, 'r') as f:
            icd_omim_mapping = json.load(f)

    return icd_omim_mapping

def get_icd_hpo_silver_mapping():
    icd2omim = get_icd_omim_mapping()
    omim2hpo = get_omim_hpo_mapping()
    icd2hpo = dict()
    for icd in icd2omim:
        for omim in icd2omim[icd]:
            if omim in omim2hpo:
                if icd not in icd2hpo:
                    icd2hpo[icd] = set()
                icd2hpo[icd].update(omim2hpo[omim])
    return icd2hpo

# if root_node == 'HP:0000118'
# only the direct children nodes of 'HP:0000118' will be considered in all annotation
def get_icd_hpo_in_limited_hpo_set(root_node=None):
    hpo_data = get_hpo4dataset()
    # po_onto = load_hpo_ontology()
    # seed_node = hpo_onto[root_node]["relations"].get("can_be", [])
    seed_node = hpo_limited_list
    print(seed_node, len(seed_node))

    node_mapping = dict()
    for s in seed_node:
        successors = hpo_data[s]["children_node"]
        for c in successors:
            # assert c not in node_mapping
            if c not in node_mapping:
                node_mapping[c] = set()
            node_mapping[c].add(s)

    icd2hpo = get_icd_hpo_silver_mapping()
    new_icd2hpo = dict()

    for icd in icd2hpo:
        new_icd2hpo[icd] = set()
        for hpo in icd2hpo[icd]:
            if hpo in node_mapping:
                new_icd2hpo[icd].update(node_mapping[hpo])

    return new_icd2hpo

def get_sentence_list_mimic(mimic_corpus_data):
    sentences_list = list()
    for doc in tqdm(mimic_corpus_data):
        if len(doc) == 0:
            continue
        lines = doc.split("\n")
        sentences_list += [l for l in lines if len(l) > 0]
    return sentences_list


if __name__ == "__main__":
    ## the code here is used to test each function
    # load_hpo_ontology()
    # plot_hpo_ontology(data)
    # hpo_desc = load_hpo_description()
    # hpo_desc = convert_hpo_to_flat_description()
    # load_mimic()
    # mimic_data, hpo_data = get_corpus(rebuild=True)
    mimic_data, hpo_data = get_corpus(rebuild=False)
    for hpo in hpo_limited_list:
        print(hpo, hpo_data[hpo])
    # print(mimic_data[0])
    # print(hpo_data['HP:0001804'])
    # for d in data[-50000:]:
    #     if '[UNK]' in d:
    #        print(d)
    # print(data[-50000:-49998])
    # print(len(data))
    # doc_list = convert_hpo_to_docs()

    # get_disease2hpo_mapping()
    # get_icd2hpo_mapping()

    # hpodata = get_hpo4dataset()
    # test_len = []
    # train_len = []
    # for key in hpodata:
    #     if len(hpodata[key]["mimic_test"]) > 0:
    #         test_len.append(len(hpodata[key]['mimic_test']))
    #     if len(hpodata[key]["mimic_train"]) > 0:
    #         train_len.append(len(hpodata[key]['mimic_train']))
    # print("MIMIC linked HPO: train %d, test: %d, total %d" % (len(train_len), len(test_len), len(hpodata.keys())))
    # print("MIMIC linked HPO: avgtrain %d, avgtest: %d" % (np.mean(train_len), np.mean(test_len)))

    ## check if matched
    # mimic_data = load_mimic()
    # print(mimic_data.loc[mimic_data.shape[0] - 1]['TEXT'][:1000])
    # print("=====")
    # print(mimic_data.loc[mimic_data.shape[0] - 1]['CLEAN_TEXT'][:1000])

    # omim_hpo = get_omim_hpo_mapping()
    # icd2omim = get_icd_omim_mapping()

    # icd2hpo = get_icd_hpo_silver_mapping()
    # hpo_subset = set()
    # for icd in icd2hpo:
    #     hpo_subset.update(icd2hpo[icd])
    # print(hpo_subset)
    # print(len(icd2hpo))
    # print(len(hpo_subset))

    # new_icd2hpo = get_icd_hpo_in_limited_hpo_set("HP:0000118")
    # print(new_icd2hpo)
    # print(len(new_icd2hpo)) # 98
    # hpo_set = set()
    # for icd in new_icd2hpo:
    #     hpo_set.update(new_icd2hpo[icd])
    # print(len(hpo_set)) # 24
    # print(hpo_set)
    # print(np.mean([len(new_icd2hpo[icd]) for icd in new_icd2hpo])) # 8.102

    # TODO: write a HPCC version of data processing
    pass
