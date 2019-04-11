import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import dataloader
import config

silver_standard_results_file = config.outputs_results_dir + "silver_standard.csv"
keyword_search_results_file = config.outputs_results_dir + "keyword_search.csv"

def silver_standard():

    #  if not os.path.exists(silver_standard_results_file):
    if True:
        mimic_data = dataloader.load_mimic()
        icd2omim = dataloader.get_icd_omim_mapping()
        omim2hpo = dataloader.get_omim_hpo_mapping()

        print("Max OMIM per ICD: %.f" % np.max([len(icd2omim[icd]) for icd in icd2omim]))
        print("Q95 OMIM per ICD: %.f" % np.quantile([len(icd2omim[icd]) for icd in icd2omim], 0.95))
        print("Q90 OMIM per ICD: %.f" % np.quantile([len(icd2omim[icd]) for icd in icd2omim], 0.90))
        print("Avg OMIM per ICD: %.f" % np.mean([len(icd2omim[icd]) for icd in icd2omim]))
        print("Med OMIM per ICD: %.f" % np.median([len(icd2omim[icd]) for icd in icd2omim]))
        print("Min OMIM per ICD: %.f" % np.min([len(icd2omim[icd]) for icd in icd2omim]))
        print("---")
        print("Max HPO per OMIM: %.f" % np.max([len(omim2hpo[omim]) for omim in omim2hpo]))
        print("Q95 HPO per OMIM: %.f" % np.quantile([len(omim2hpo[omim]) for omim in omim2hpo], 0.95))
        print("Q90 HPO per OMIM: %.f" % np.quantile([len(omim2hpo[omim]) for omim in omim2hpo], 0.90))
        print("Avg HPO per OMIM: %.f" % np.mean([len(omim2hpo[omim]) for omim in omim2hpo]))
        print("Med HPO per OMIM: %.f" % np.median([len(omim2hpo[omim]) for omim in omim2hpo]))
        print("Min HPO per OMIM: %.f" % np.min([len(omim2hpo[omim]) for omim in omim2hpo]))

        # icd2hpo = dataloader.get_icd_hpo_silver_mapping()
        icd2hpo = dataloader.get_icd_hpo_in_limited_hpo_set(dataloader.hpo_phenotypic_abnormality_id)
        print("---")
        print("Max HPO per ICD: %.f" % np.max([len(icd2hpo[icd]) for icd in icd2hpo]))
        print("Q95 HPO per ICD: %.f" % np.quantile([len(icd2hpo[icd]) for icd in icd2hpo], 0.95))
        print("Q90 HPO per ICD: %.f" % np.quantile([len(icd2hpo[icd]) for icd in icd2hpo], 0.90))
        print("Avg HPO per ICD: %.f" % np.mean([len(icd2hpo[icd]) for icd in icd2hpo]))
        print("Med HPO per ICD: %.f" % np.median([len(icd2hpo[icd]) for icd in icd2hpo]))
        print("Min HPO per ICD: %.f" % np.min([len(icd2hpo[icd]) for icd in icd2hpo]))

        def icd2hpo_mapping(icdstr):
            icd_list = icdstr.split("/")
            hposet = set()
            for icd in icd_list:
                if icd in icd2hpo:
                    hposet.update(icd2hpo[icd])
            hpostr = "/".join(hposet)
            return hpostr

        print("Computing silver standard ...")
        mimic_data["HPO_CODE_LIST"] = mimic_data['ICD9_CODE_LIST'].apply(icd2hpo_mapping)

        mimic_data = mimic_data[["ICD9_CODE_LIST", "HPO_CODE_LIST", "CLEAN_TEXT"]]

        mimic_data.to_csv(silver_standard_results_file)

    else:
        mimic_data = pd.read_csv(silver_standard_results_file)

    return mimic_data

def keyword_search():

    if not os.path.exists(keyword_search_results_file):
    # if True:

        print("Keyword searching ...")
        hpodata = dataloader.get_hpo4dataset()

        sentence_dict = dict()
        sentence_list = list()
        hpo_dict = dict()
        hpo_list = list()

        print("Creating sentence list ...")
        # for training only
        # create a complete index for all sentences
        for hpoid in tqdm(hpodata):
            for sentence in (hpodata[hpoid]['mimic_train'] | hpodata[hpoid]['mimic_test']):
                if sentence in sentence_dict:
                    sentence_id = sentence_dict[sentence]
                    # sentence_list[sentence_id]['hpo_ids'].add(hpoid)
                    continue
                assert sentence.replace("[UNK]", "").islower()
                assert len(sentence) > 0
                assert sentence not in sentence_dict
                sentence_dict[sentence] = len(sentence_list)
                sentence_list.append({
                    "sentence": sentence,
                    # "hpo_ids": {hpoid}
                    "hpo_ids": set()
                })

        def _dfs(hpoid, depth):

            if hpoid in hpo_dict:
                return

            if hpodata[hpoid]['status'] == True:
                assert hpoid not in hpo_dict
                hpo_dict[hpoid] = len(hpo_list)
                hpo_list.append({
                    "hpo_id": hpoid,
                    "sentences_id": set([sentence_dict[s] for s in (hpodata[hpoid]['mimic_train'] | hpodata[hpoid]['mimic_test'])])
                })
                # sentences of all children nodes should be included as well
                for child in hpodata[hpoid]['children_node']:
                    _dfs(child, depth + 1)
                    hpo_list[hpo_dict[hpoid]]["sentences_id"] |= hpo_list[hpo_dict[child]]["sentences_id"]
            else:
                for child in hpodata[hpoid]['children_node']:
                    _dfs(child, depth + 1)

        print("Searching on HPO hierarchy ...")
        _dfs(dataloader.hpo_root_id, depth=0)

        print("Updating HPO <-> Sentence ...")
        # construct relations sentences -> HPO
        for hpo_sample in tqdm(hpo_list):
            hpo_id = hpo_sample['hpo_id']
            for sentence_id in hpo_sample['sentences_id']:
                sentence_list[sentence_id]["hpo_ids"].add(hpo_id)

        sentence2hpo_with_parent = dict()
        sentence2hpo_without_parent = dict()

        print("Finalizing HPO <-> Sentence ...")
        for hpoid in tqdm(hpodata):
            sentences = (hpodata[hpoid]['mimic_train'] | hpodata[hpoid]['mimic_test'])
            for sentence in sentences:
                if sentence not in sentence2hpo_with_parent:
                    sentence2hpo_with_parent[sentence] = set()
                sentence_id = sentence_dict[sentence]
                assert sentence == sentence_list[sentence_id]['sentence']
                sentence2hpo_with_parent[sentence].update(sentence_list[sentence_id]['hpo_ids'])
                if sentence not in sentence2hpo_without_parent:
                    sentence2hpo_without_parent[sentence] = set()
                sentence2hpo_without_parent[sentence].add(hpoid)

        def text_to_hpo_by_keyword_with_parent(text):
            hposet = set()
            sentences = text.split("\n")
            for sentence in sentences:
                if sentence in sentence2hpo_with_parent:
                    hposet.update(sentence2hpo_with_parent[sentence])
            hpostr = "/".join(hposet)
            return hpostr

        def text_to_hpo_by_keyword_without_parent(text):
            hposet = set()
            sentences = text.split("\n")
            for sentence in sentences:
                if sentence in sentence2hpo_without_parent:
                    hposet.update(sentence2hpo_without_parent[sentence])
            hpostr = "/".join(hposet)
            return hpostr

        mimic_data = dataloader.load_mimic()

        print("Computing baseline: keyword search ...")
        mimic_data['HPO_CODE_LIST_KEYWORD_SEARCH_WITH_PARENT'] = mimic_data['CLEAN_TEXT'].apply(text_to_hpo_by_keyword_with_parent)
        mimic_data['HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT'] = mimic_data['CLEAN_TEXT'].apply(text_to_hpo_by_keyword_without_parent)

        mimic_data = mimic_data[["ICD9_CODE_LIST",
                                 "HPO_CODE_LIST_KEYWORD_SEARCH_WITH_PARENT",
                                 "HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT",
                                 "CLEAN_TEXT"]]
        mimic_data.to_csv(keyword_search_results_file)

    else:
        mimic_data = pd.read_csv(keyword_search_results_file)

    return mimic_data

def keyword_search_with_negation():
    # TODO: baseline: keyword search with negation
    pass

def random_pick():
    # TODO: randomly pick a random number of HPO terms
    pass

def topic_model():
    # TODO: use topic model and design a rule to connect topics with HPO
    pass

if __name__ == '__main__':

    mimic_data = silver_standard()
    hpo_list = mimic_data["HPO_CODE_LIST"].tolist()
    print("Num of EHR has HPO %d/%d" % (np.sum([1 for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]), len(hpo_list)))
    print("Avg HPO for all %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Median HPO for all %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Avg HPO for those have %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    print("Median HPO for those have %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    # Num of EHR has HPO 27980/52722
    # Avg HPO for all 8
    # Median HPO for all 5
    # Avg HPO for those have 15
    # Median HPO for those have 12

    '''
    mimic_data = keyword_search()
    hpo_list = mimic_data["HPO_CODE_LIST_KEYWORD_SEARCH_WITH_PARENT"].tolist()
    print("Num of EHR has HPO %d/%d" % (np.sum([1 for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]), len(hpo_list)))
    print("Avg HPO for all %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Median HPO for all %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Avg HPO for those have %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    print("Median HPO for those have %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    '''
    # Num of EHR has HPO 52721/52722
    # Avg HPO for all 128
    # Median HPO for all 125
    # Avg HPO for those have 128
    # Median HPO for those have 125
    '''
    hpo_list = mimic_data["HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT"].tolist()
    print("Num of EHR has HPO %d/%d" % (np.sum([1 for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]), len(hpo_list)))
    print("Avg HPO for all %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Median HPO for all %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Avg HPO for those have %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    print("Median HPO for those have %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    '''
    # Num of EHR has HPO 52721/52722
    # Avg HPO for all 44
    # Median HPO for all 41
    # Avg HPO for those have 44
    # Median HPO for those have 41


    pass

