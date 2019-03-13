# Dataloader
# Data loading and early analysis on data

import os
import re
import json
import pronto
import pandas as pd

import config

# environment variables, subject to changes according to local environment
# the folder that contains original data
data_dir = "../data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# the path to HPO ontology file
# download link: https://hpo.jax.org/app/download/ontology
hpo_ontology_file = data_dir + "HPO/hp.owl"
# the path to MIMIC-III files
# download link: https://mimic.physionet.org/
# application to access the data required
mimic_note_file = data_dir + "MIMIC/NOTEEVENTS.csv"
mimic_diag_file = data_dir + "MIMIC/DIAGNOSES_ICD.csv"

# intermediate files which will be created
# created by load_hpo_description()
hpo_description_file = data_dir + "HPO/description.json"
# created by load_mimic()
mimic_file = data_dir + "MIMIC/ready.csv"
# created get_corpus()
text_corpus_file = data_dir + "corpus.txt"
# created get_vocab()
vocab_file = data_dir + "vocab.txt"

def load_hpo_ontology():
    ont = pronto.Ontology(hpo_ontology_file)
    data = json.loads(ont.json)
    return data

def load_hpo_description():
    import queue

    if not os.path.exists(hpo_description_file):

        print("Loading HPO description ...")

        domains = [
            'name',
            'desc',
            'other/comment',
            'other/hasExactSynonym',
            'other/hasRelatedSynonym'
        ]

        data = load_hpo_ontology()
        process_data = dict()
        start_id = "HP:0000001"

        node_queue = queue.Queue()
        node_queue.put(start_id)

        while not node_queue.empty():
            node = node_queue.get()
            process_data[node] = dict()
            for d in domains:
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
            json.dump(data, outfile)

        print("HPO descriptions have been extracted and save to %s" % hpo_description_file)

    else:

        print("Loading HPO descriptions from file %s" % hpo_description_file)

        with open(hpo_description_file, 'r') as infile:
            data = json.load(infile)

    return data

def plot_hpo_ontology():

    data = load_hpo_ontology()

    from pyvis.network import Network
    import queue
    net = Network()

    # HP:0000001 All
    # HP:0000018 Phenotypic Abnormality
    # Other branch is also necessary
    start_id = "HP:0000001"

    node_queue = queue.Queue()
    node_queue.put(start_id)
    net.add_node(start_id)

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

# TODO include hpo descirption into the corpus and vocab
def get_corpus(most_common=config.vocabulary_size, sentence_length=32, rebuild=False):

    from collections import Counter
    from tqdm import tqdm
    counter = Counter()

    if not os.path.exists(vocab_file) or not os.path.exists(text_corpus_file) or rebuild:
        mimic_data = load_mimic()
        text_data = mimic_data["TEXT"].tolist()

        # import spacy
        # nlp = spacy.load("en_core_web_sm")

        white_space = [" ", "\n", "\r", "\t", "\f", "\v"]
        special_token_to_ignore = ["?"]

        doc_list = []
        print("Processing the original data and creating vocabulary ...")
        for doc in tqdm(text_data[:]):
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
                # if text != "" and text not in special_token_to_ignore:
                if text != "":
                    counter[text] += 1
                tlist.append(text)
            doc_list.append(tlist)

        word_counts = [x for x in counter.items() if x[1] >= 1]
        word_counts.sort(key=lambda x: x[1], reverse=True)
        word_counts = word_counts[:most_common + 1]
        word_counts = dict([(w[0], w[1]) for w in word_counts])
        for t in special_token_to_ignore + white_space:
            assert t not in word_counts

        with open(vocab_file, 'w') as out_vocab_file:
            for word in word_counts:
                out_vocab_file.write(word + "\n")
        print("Vocab saved to %s" % vocab_file)

        new_doc_list = []
        print("Cleaning the original data ...")
        for doc in tqdm(doc_list[:]):
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

            new_doc_list.append(new_doc)

        with open(text_corpus_file, 'w') as out_corpus_file:
            for doc in new_doc_list:
                out_corpus_file.write(doc + "==DOCSPLIT==\n")
        print("Corpus saved to %s" % text_corpus_file)

        text_data = new_doc_list

    else:
        with open(text_corpus_file, 'r') as infile:
            text_data = infile.read()
        print("Corpus loaded from %s" % text_corpus_file)

        text_data = text_data.split("==DOCSPLIT==\n")

    # TODO: return vocab
    return text_data

if __name__ == "__main__":
    # load_hpo_ontology()
    # plot_hpo_ontology(data)
    # load_hpo_description()
    # load_mimic()
    data = get_corpus()
    print(data[0])
    print(len(data))
