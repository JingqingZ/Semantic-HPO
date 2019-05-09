import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import dataloader
import config
import mimic_proc

silver_standard_results_file = config.outputs_results_dir + "silver_standard.csv"
keyword_search_results_file = config.outputs_results_dir + "keyword_search.csv"
ehr_phenolyzer_ncbo_results_file = config.outputs_results_dir + "ehr_pheno_ncbo.csv"

def silver_standard():

    if not os.path.exists(silver_standard_results_file):
    # if True:
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
        # Max OMIM per ICD: 63                                                                                                  │·····················································································
        # Q95 OMIM per ICD: 13                                                                                                  │·····················································································
        # Q90 OMIM per ICD: 9                                                                                                   │·····················································································
        # Avg OMIM per ICD: 4                                                                                                   │·····················································································
        # Med OMIM per ICD: 2                                                                                                   │·····················································································
        # Min OMIM per ICD: 1                                                                                                   │·····················································································
        # ---                                                                                                                   │·····················································································
        # Max HPO per OMIM: 127                                                                                                 │·····················································································
        # Q95 HPO per OMIM: 37                                                                                                  │·····················································································
        # Q90 HPO per OMIM: 28                                                                                                  │·····················································································
        # Avg HPO per OMIM: 13                                                                                                  │·····················································································
        # Med HPO per OMIM: 9                                                                                                   │·····················································································
        # Min HPO per OMIM: 1

        # icd2hpo = dataloader.get_icd_hpo_silver_mapping()
        icd2hpo = dataloader.get_icd_hpo_in_limited_hpo_set(dataloader.hpo_phenotypic_abnormality_id)
        print("---")
        print("Max HPO per ICD: %.f" % np.max([len(icd2hpo[icd]) for icd in icd2hpo]))
        print("Q95 HPO per ICD: %.f" % np.quantile([len(icd2hpo[icd]) for icd in icd2hpo], 0.95))
        print("Q90 HPO per ICD: %.f" % np.quantile([len(icd2hpo[icd]) for icd in icd2hpo], 0.90))
        print("Avg HPO per ICD: %.f" % np.mean([len(icd2hpo[icd]) for icd in icd2hpo]))
        print("Med HPO per ICD: %.f" % np.median([len(icd2hpo[icd]) for icd in icd2hpo]))
        print("Min HPO per ICD: %.f" % np.min([len(icd2hpo[icd]) for icd in icd2hpo]))
        # Max HPO per ICD: 20                                                                                                   │·····················································································
        # Q95 HPO per ICD: 16                                                                                                   │·····················································································
        # Q90 HPO per ICD: 14                                                                                                   │·····················································································
        # Avg HPO per ICD: 8                                                                                                    │·····················································································
        # Med HPO per ICD: 8                                                                                                    │·····················································································
        # Min HPO per ICD: 0

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
            # sentences = text.split("\n")
            sentences = mimic_proc.get_sentences_from_mimic(text)
            for sentence in sentences:
                if sentence in sentence2hpo_with_parent:
                    hposet.update(sentence2hpo_with_parent[sentence])
            hpostr = "/".join(hposet)
            return hpostr

        def text_to_hpo_by_keyword_without_parent(text):
            hposet = set()
            # sentences = text.split("\n")
            sentences = mimic_proc.get_sentences_from_mimic(text)
            for sentence in sentences:
                if sentence in sentence2hpo_without_parent:
                    hposet.update(sentence2hpo_without_parent[sentence])
            hpostr = "/".join(hposet)
            return hpostr

        mimic_data = dataloader.load_mimic()

        print("Computing baseline: keyword search ...")
        mimic_data['HPO_CODE_LIST_KEYWORD_SEARCH_WITH_PARENT'] = mimic_data['CLEAN_TEXT'].apply(text_to_hpo_by_keyword_with_parent)
        mimic_data['HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT'] = mimic_data['CLEAN_TEXT'].apply(text_to_hpo_by_keyword_without_parent)

        # root_node = dataloader.hpo_phenotypic_abnormality_id
        # hpo_onto = dataloader.load_hpo_ontology()
        # hpo_predecessors_node = hpo_onto[root_node]["relations"].get("can_be", [])
        # new_icd2hpo = dataloader.get_icd_hpo_in_limited_hpo_set(dataloader.hpo_phenotypic_abnormality_id)
        # hpo_predecessors_node = set()
        # for icd in new_icd2hpo:
        #     hpo_predecessors_node.update(new_icd2hpo[icd])
        hpo_predecessors_node = set(dataloader.hpo_limited_list)

        def keep_predecessors_only(text):
            if not isinstance(text, str) or len(text) == 0:
                return ''
            hpolist = text.split("/")
            nhpolist = [hpo for hpo in hpolist if hpo in hpo_predecessors_node]
            hpostr = "/".join(nhpolist)
            return hpostr

        mimic_data['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'] = mimic_data['HPO_CODE_LIST_KEYWORD_SEARCH_WITH_PARENT'].apply(keep_predecessors_only)

        mimic_data = mimic_data[["ICD9_CODE_LIST",
                                 "HPO_CODE_LIST_KEYWORD_SEARCH_WITH_PARENT",
                                 "HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT",
                                 "HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY",
                                 "CLEAN_TEXT"]]
        mimic_data.to_csv(keyword_search_results_file)

    else:
        mimic_data = pd.read_csv(keyword_search_results_file)

    return mimic_data

def keyword_search_with_negation():

    keyword_search_df = keyword_search()

    if "HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY_WITH_NEGATION" not in keyword_search_df.columns:

        forward_negative_word = [
            'not', 'never', 'hardly', 'rarely', 'rule out', 'rules out',
            'ruled out', 'no', 'no evidence of', 'no support of', 'no suspicion of',
            'without', 'without evidence of', 'without support of', 'without suspicion of',
            'deny', 'denies', 'denied', 'unlikely', 'no change to',
            'no signs of', 'without signs of', 'absence of', 'doubt', 'doubts', 'doubted',
            'no further', 'without any further', 'without further'
        ]
        backward_negative_word = [
            'not', 'never', 'hardly', 'rarely', 'is rule out',
            'has been ruled out', 'has no evidence', 'has no support', 'has no suspicion',
            'is denied', 'is unlikely',
            'has no signs', 'is absent', 'is not demonstrated', 'is negative', 'is doubted',
        ]
        hpo_list = keyword_search_df["HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT"].tolist()

        refer = keyword_search_df["HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY"].tolist()

        clean_text = keyword_search_df["CLEAN_TEXT"].tolist()
        hpo_data = dataloader.get_hpo4dataset()
        assert len(hpo_list) == len(clean_text)

        hpo_children_info = dataloader.get_hpo_children_info()
        children_to_predecessor_mapping = dict()
        for hpo in dataloader.hpo_limited_list:
            children = hpo_children_info[hpo]
            for cnode in children:
                if cnode not in children_to_predecessor_mapping:
                    children_to_predecessor_mapping[cnode] = set()
                children_to_predecessor_mapping[cnode].add(hpo)

        num_of_nega = 0
        new_hpo_list = []
        for i in trange(len(hpo_list)):
            hpostr = hpo_list[i]
            doc = clean_text[i]
            if isinstance(hpostr, float):
                new_hpo_list.append("")
                continue
            hpos = hpostr.split("/")
            hposet = set()
            for hpo in hpos:
                terms = hpo_data[hpo]['terms']
                nega_terms = []
                for term in terms:
                    for pre in forward_negative_word:
                        nega_terms.append(pre + " " + term)
                    for pos in backward_negative_word:
                        nega_terms.append(term + " " + pos)
                nega_flag = False
                for term in nega_terms:
                    if term in doc:
                        nega_flag = True
                        num_of_nega += 1
                        break
                if not nega_flag:
                    if hpo in children_to_predecessor_mapping:
                        hposet.update(children_to_predecessor_mapping[hpo])
            new_hpo_list.append("/".join(hposet))
        assert len(new_hpo_list) == len(hpo_list)
        print("Num of negation detected: %d" % num_of_nega)

        keyword_search_df["HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY_WITH_NEGATION"] = new_hpo_list
        keyword_search_df.to_csv(keyword_search_results_file)

    return keyword_search_df

def random_pick():
    # TODO: randomly pick a random number of HPO terms
    pass

def topic_model():
    # TODO: use topic model and design a rule to connect topics with HPO
    pass

def ehr_phenolyzer():

    # if not os.path.exists(ehr_phenolyzer_ncbo_results_file):
    if True:
        import sys
        PACKAGE_PARENT = '../baselines/EHR-Phenolyzer/'
        SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
        sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

        # import ehr_phenolyzer
        import lib.pyncbo_annotator as ncbo_ann

        # corpus_mimic, _ = dataloader.get_corpus()
        # corpus_mimic = corpus_mimic[:5]
        # del _
        full_mimic_data = dataloader.load_mimic()
        # assert len(corpus_mimic) == full_mimic_data.shape[0]

        children_info = dataloader.get_hpo_children_info()
        children_to_predecessor = dict()
        for hpo_id in dataloader.hpo_limited_list:
            children = children_info[hpo_id]
            for cnode in children:
                if cnode not in children_to_predecessor:
                    children_to_predecessor[cnode] = set()
                children_to_predecessor[cnode].add(hpo_id)

        def _convert_to_predecessor(hpo_list):
            if isinstance(hpo_list, float):
                return ""
            new_hpo_set = set()
            for hpo_id in hpo_list.split("/"):
                if hpo_id not in children_to_predecessor:
                    continue
                new_hpo_set.update(children_to_predecessor[hpo_id])
            return "/".join(new_hpo_set)

        def _ncbo_annotater(text):
            text = text.replace("\n", " ")
            hpo_list = ncbo_ann.run_ncbo_annotator(text)
            hpostr = "/".join(hpo_list)
            return hpostr

        print("Computing baseline: EHR Phenolyzer NCBO ...")
        group_num = 100
        new_mimic_data = None
        for i in trange(full_mimic_data.shape[0] // group_num + 1):

            if i == full_mimic_data.shape[0] // group_num:
                if full_mimic_data.shape[0] == group_num * i:
                    break
                mimic_data = full_mimic_data[i * group_num : ].copy(deep=True)
                # list_hpo_results = ehr_phenolyzer.annotate(corpus_mimic[i * group_num : ], nlp_mode='NCBO')
            else:
                mimic_data = full_mimic_data[i * group_num : (i + 1) * group_num].copy(deep=True)
                # list_hpo_results = ehr_phenolyzer.annotate(corpus_mimic[i * group_num : (i + 1) * group_num], nlp_mode='NCBO')

            # assert mimic_data.shape[0] == len(list_hpo_results)

            # mimic_data['HPO_CODE_LIST_EHR_PHENO'] = pd.Series(["/".join(l) for l in list_hpo_results])
            mimic_data['HPO_CODE_LIST_EHR_PHENO'] = mimic_data['CLEAN_TEXT'].apply(_ncbo_annotater)
            mimic_data['HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY'] = mimic_data['HPO_CODE_LIST_EHR_PHENO'].apply(_convert_to_predecessor)

            mimic_data = mimic_data[["ICD9_CODE_LIST",
                                     "HPO_CODE_LIST_EHR_PHENO",
                                     "HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY",
                                     "CLEAN_TEXT"]]
            if new_mimic_data is None:
                new_mimic_data = mimic_data
            else:
                new_mimic_data = pd.concat([new_mimic_data, mimic_data])

            new_mimic_data.to_csv(ehr_phenolyzer_ncbo_results_file)

        # print(new_mimic_data["HPO_CODE_LIST_EHR_PHENO"])
        # print(new_mimic_data.shape)

    else:
        new_mimic_data = pd.read_csv(ehr_phenolyzer_ncbo_results_file)

    return new_mimic_data

if __name__ == '__main__':

    '''
    mimic_data = silver_standard()
    hpo_list = mimic_data["HPO_CODE_LIST"].tolist()
    print("Num of EHR has HPO %d/%d" % (np.sum([1 for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]), len(hpo_list)))
    print("Avg HPO for all %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Median HPO for all %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Avg HPO for those have %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    print("Median HPO for those have %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    '''
    # ALL HPO
    # Num of EHR has HPO 27980/52722
    # Avg HPO for all 8
    # Median HPO for all 5
    # Avg HPO for those have 15
    # Median HPO for those have 12
    # LIMITED HPO (direct children of HP:0000118)
    # Num of EHR has HPO 27980/52722                                                                                        │·····················································································
    # Avg HPO for all 3                                                                                                     │·····················································································
    # Median HPO for all 4                                                                                                  │·····················································································
    # Avg HPO for those have 6                                                                                              │·····················································································
    # Median HPO for those have 5

    '''
    mimic_data = keyword_search()
    hpo_list = mimic_data['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'].to_list()
    print("Num of EHR has HPO %d/%d" % (np.sum([1 for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]), len(hpo_list)))
    print("Avg HPO for all %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Median HPO for all %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Avg HPO for those have %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    print("Median HPO for those have %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    '''
    # Avg HPO for all 9
    # Median HPO for all 9
    # Avg HPO for those have 9
    # Median HPO for those have 9

    '''
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

    '''
    mimic_data = keyword_search_with_negation()
    hpo_list = mimic_data['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY_WITH_NEGATION'].to_list()
    print("Num of EHR has HPO %d/%d" % (np.sum([1 for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]), len(hpo_list)))
    print("Avg HPO for all %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Median HPO for all %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) if isinstance(hstr, str) else 0 for hstr in hpo_list]))
    print("Avg HPO for those have %.f" % np.mean([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    print("Median HPO for those have %.f" % np.median([len([h for h in hstr.split("/") if len(h) > 0]) for hstr in hpo_list if not isinstance(hstr, float) and len(hstr) > 0]))
    '''
    # Num of EHR has HPO 52721/52722
    # Avg HPO for all 11
    # Median HPO for all 11
    # Avg HPO for those have 11
    # Median HPO for those have 11

    mimic_data = ehr_phenolyzer()

    pass

