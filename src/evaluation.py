import numpy as np
from tqdm import tqdm, trange

import dataloader
import config
import baselines
import decision
import utils

_set_of_hpos = set(dataloader.hpo_limited_list)
_global_print_best_worst_case = False

def precision(set1, set2):
    counter = 0
    for s in set2:
        if s in set1:
            counter += 1
    return counter / len(set2)

def recall(set1, set2):
    counter = 0
    for s in set1:
        if s in set2:
            counter += 1
    return counter / len(set1)

# reference: https://en.wikipedia.org/wiki/Overlap_coefficient
def overlap_coefficient(set1, set2):
    return len(set1 & set2) / min(len(set1), len(set2))

# reference: https://en.wikipedia.org/wiki/Jaccard_index
def jaccard(set1, set2):
   return len(set1 & set2) / len(set1 | set2)

def _evaluate(list_of_set1, list_of_set2, func):

    overlap_list = list()
    at_least_one_match_list = list()

    assert len(list_of_set1) == len(list_of_set2)

    score_dict = dict()
    for i in range(len(list_of_set1)):
        if isinstance(list_of_set1[i], float) or isinstance(list_of_set2[i], float):
            continue
        hposet1 = set([_ for _ in list_of_set1[i].split("/") if len(_) > 0])
        hposet2 = set([_ for _ in list_of_set2[i].split("/") if len(_) > 0])

        # for hpo in (hposet1 | hposet2):
        #     assert hpo in _set_of_hpos

        if len(hposet1) > 0 and len(hposet2) > 0:
            overlap = func(hposet1, hposet2)
            at_least_one_match = 1 if len(hposet1 & hposet2) > 0 else 0
            overlap_list.append(overlap)
            at_least_one_match_list.append(at_least_one_match)

            score_dict[i] = overlap

    if _global_print_best_worst_case:

        sorted_score = sorted(score_dict.items(), key=lambda kv: kv[1])

        # worst case
        for i in range(10):
            print(i, sorted_score[i])
            print(sorted(list_of_set1[sorted_score[i][0]].split("/")))
            print(sorted(list_of_set2[sorted_score[i][0]].split("/")))
            print("---")

        print("====")

        # best case
        for i in range(1, 31):
            print(-i, sorted_score[-i])
            print(sorted(list_of_set1[sorted_score[-i][0]].split("/")))
            print(sorted(list_of_set2[sorted_score[-i][0]].split("/")))
            print("---")

    # print("Total num of records: %d" % len(list_of_set1))
    # print("Total num of valid comparison: %d" % len(overlap_list))
    print("Avg %s index: %.8f" % (func.__name__, np.mean(overlap_list)))
    print("Median %s index: %.8f" % (func.__name__, np.median(overlap_list)))
    # print("Avg at least one matched: %.8f" % np.mean(at_least_one_match_list))
    # print("Median at least one matched: %.8f" % np.median(at_least_one_match_list))

    return score_dict, np.mean(overlap_list)

def evaluate_keyword_search(column_of_keyword, func, mode='all'):

    silver_data = baselines.silver_standard()
    keyword_data = baselines.keyword_search()

    silver = silver_data['HPO_CODE_LIST'].tolist()
    keyword = keyword_data[column_of_keyword].tolist()
    assert len(silver) == len(keyword)
    assert len(silver) == config.total_num_mimic_record

    assert silver_data['CLEAN_TEXT'].tolist()[348] == keyword_data['CLEAN_TEXT'].tolist()[348]

    if mode == 'train':
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [keyword[index] for index in config.mimic_train_indices],
            func=func
        )
    elif mode == 'test':
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [keyword[index] for index in config.mimic_test_indices],
            func=func
        )
    elif mode == 'all':
        _evaluate(
            silver, keyword,
            func=func
        )
    elif mode == 'complete':
        print("Training set")
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [keyword[index] for index in config.mimic_train_indices],
            func=func
        )
        print("---------")
        print("Testing set")
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [keyword[index] for index in config.mimic_test_indices],
            func=func
        )
        print("---------")
        print("Overall")
        _evaluate(
            silver, keyword,
            func=func
        )
    else:
        raise Exception('Invalid mode')

def evaluate_keyword_search_with_negation(column_of_keyword, func, mode='all'):

    silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    keyword = baselines.keyword_search_with_negation()[column_of_keyword].tolist()
    assert len(silver) == len(keyword)
    assert len(silver) == config.total_num_mimic_record

    if mode == 'train':
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [keyword[index] for index in config.mimic_train_indices],
            func=func
        )
    elif mode == 'test':
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [keyword[index] for index in config.mimic_test_indices],
            func=func
        )
    elif mode == 'all':
        _evaluate(
            silver, keyword,
            func=func
        )
    else:
        raise Exception('Invalid mode')

def get_icd2hpo_3digit_results():
    icd2limitedhpo_mapping = dataloader.get_3digit_icd_hpo_in_limited_hpo_set()

    def filter_icd(icdstr):
        if not isinstance(icdstr, str):
            return ""
        icd_list = icdstr.split("/")
        filtered_icd_list = [icd[:3] for icd in icd_list if icd[:3] in icd2limitedhpo_mapping]
        filtered_icdstr = "/".join(filtered_icd_list)
        return filtered_icdstr

    icd_code = baselines.silver_standard()['ICD9_CODE_LIST'].tolist()

    icd_code = [filter_icd(icdstr) for icdstr in icd_code]

    silver = []
    for icdstr in icd_code:
        icdlist = icdstr.split("/")
        hposet = set()
        for icd in icdlist:
            if len(icd) == 0:
                continue
            hposet.update(icd2limitedhpo_mapping[icd])
        silver.append("/".join(hposet))

    return silver

def evaluate_unsupervised_method(column_of_keyword, threshold, decision_mode, func, mode='complete', comb_mode='union'):

    silver = get_icd2hpo_3digit_results()
    # silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    # silver = get_strong_silver(comb_mode=comb_mode)
    unsuper = decision.results_of_alpha_out(threshold=threshold, mode=decision_mode)[column_of_keyword].tolist()
    utils.print_statictis_of_hpo(unsuper)
    assert len(silver) == len(unsuper)
    assert len(silver) == config.total_num_mimic_record

    '''
    if comb_mode == "intersection":
        # comb_func = lambda x, y: x & y
        func = precision
    elif comb_mode == "union":
        # comb_func = lambda x, y: x | y
        func = recall
    else:
        raise Exception("Invalid comb_mode")
    '''

    if mode == 'train':
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [unsuper[index] for index in config.mimic_train_indices],
            func=func
        )
    elif mode == 'test':
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [unsuper[index] for index in config.mimic_test_indices],
            func=func
        )
    elif mode == 'all':
        _evaluate(
            silver, unsuper,
            func=func
        )
    elif mode == "complete":
        print("Training set")
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [unsuper[index] for index in config.mimic_train_indices],
            func=func
        )
        print("---------")
        print("Testing set")
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [unsuper[index] for index in config.mimic_test_indices],
            func=func
        )
        print("---------")
        print("Overall")
        _evaluate(
            silver, unsuper,
            func=func
        )
    else:
        raise Exception('Invalid mode')

def evaluate_of_baselines(func, mode='test'):

    # silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    silver = get_icd2hpo_3digit_results()

    keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'].tolist()
    ncbo = baselines.ehr_phenolyzer_ncbo_annotator()["HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY"].tolist()
    obo = baselines.obo_annotator()["HPO_CODE_LIST_OBO_ANNO_PREDECESSORS_ONLY"].tolist()
    metamap = baselines.aggregate_metamap_results()["HPO_CODE_LIST_METAMAP_ANNO_PREDECESSORS_ONLY"].tolist()

    utils.print_statictis_of_hpo(keyword)
    utils.print_statictis_of_hpo(ncbo)
    utils.print_statictis_of_hpo(obo)
    utils.print_statictis_of_hpo(metamap)

    all_baselines = [keyword, ncbo, obo, metamap]
    name_baselines = ['keyword', 'ncbo', 'obo', 'metamap']

    for pidx, prediction in enumerate(all_baselines):

        print("===============")
        print("SILVER vs. BASELINE (%s)" % (name_baselines[pidx]))

        assert len(silver) == len(prediction)
        assert len(silver) == config.total_num_mimic_record

        if mode == 'train':
            _evaluate(
                [silver[index] for index in config.mimic_train_indices],
                [prediction[index] for index in config.mimic_train_indices],
                func=func
            )
        elif mode == 'test':
            _evaluate(
                [silver[index] for index in config.mimic_test_indices],
                [prediction[index] for index in config.mimic_test_indices],
                func=func
            )
        elif mode == 'all':
            _evaluate(
                silver, prediction,
                func=func
            )
        elif mode == "complete":
            print("Training set")
            _evaluate(
                [silver[index] for index in config.mimic_train_indices],
                [prediction[index] for index in config.mimic_train_indices],
                func=func
            )
            print("---------")
            print("Testing set")
            _evaluate(
                [silver[index] for index in config.mimic_test_indices],
                [prediction[index] for index in config.mimic_test_indices],
                func=func
            )
            print("---------")
            print("Overall")
            _evaluate(
                silver, prediction,
                func=func
            )
        else:
            raise Exception('Invalid mode')

def combine_methods(list1, list2, func):

    def _str_to_set(text):
        if not isinstance(text, str):
            return set()
        return set(text.split("/"))

    assert len(list1) == len(list2)

    result_list = []
    for idx in range(len(list1)):
        set1 = _str_to_set(list1[idx])
        set2 = _str_to_set(list2[idx])
        comb_set = func(set1, set2)
        result_list.append("/".join(comb_set))

    return result_list

def evaluate_ncbo_annotator(column_of_results="HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY", func=jaccard, mode='complete'):

    # silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'].tolist()
    ncbo = baselines.ehr_phenolyzer_ncbo_annotator()[column_of_results].tolist()
    func = precision

    silver = combine_methods(keyword, ncbo, func=lambda x, y: x & y)

    assert len(silver) == len(ncbo)
    assert len(silver) == config.total_num_mimic_record

    if mode == 'train':
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [ncbo[index] for index in config.mimic_train_indices],
            func=func
        )
    elif mode == 'test':
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [ncbo[index] for index in config.mimic_test_indices],
            func=func
        )
    elif mode == 'all':
        _evaluate(
            silver, ncbo,
            func=func
        )
    elif mode == "complete":
        print("Training set")
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [ncbo[index] for index in config.mimic_train_indices],
            func=func
        )
        print("---------")
        print("Testing set")
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [ncbo[index] for index in config.mimic_test_indices],
            func=func
        )
        print("---------")
        print("Overall")
        _evaluate(
            silver, ncbo,
            func=func
        )
    else:
        raise Exception('Invalid mode')

def get_strong_silver(keyword=None, ncbo=None, obo=None, comb_mode="union"):

    if comb_mode == "intersection":
        comb_func = lambda x, y: x & y
    elif comb_mode == "union":
        comb_func = lambda x, y: x | y
    else:
        raise Exception("Invalid comb_mode")

    print("Loading a strong silver standard from baselines (%s)" % comb_mode)
    if keyword is None:
        keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'].tolist()
    if ncbo is None:
        ncbo = baselines.ehr_phenolyzer_ncbo_annotator()["HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY"].tolist()
    if obo is None:
        obo = baselines.obo_annotator()["HPO_CODE_LIST_OBO_ANNO_PREDECESSORS_ONLY"].tolist()

    silver = combine_methods(keyword, obo, func=comb_func)
    silver = combine_methods(silver, ncbo, func=comb_func)

    return silver

def evaluate_of_baselines_combination(mode='test', comb_mode="union"):

    # silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'].tolist()
    ncbo = baselines.ehr_phenolyzer_ncbo_annotator()["HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY"].tolist()
    obo = baselines.obo_annotator()["HPO_CODE_LIST_OBO_ANNO_PREDECESSORS_ONLY"].tolist()

    all_baselines = [keyword, ncbo, obo]
    name_baselines = ['keyword', 'ncbo', 'obo']
    # func = precision

    if comb_mode == "intersection":
        # comb_func = lambda x, y: x & y
        func = precision
    elif comb_mode == "union":
        # comb_func = lambda x, y: x | y
        func = recall
    else:
        raise Exception("Invalid comb_mode")

    silver = get_strong_silver(keyword, ncbo, obo, comb_mode)

    for pidx, prediction in enumerate(all_baselines):

        print("===============")
        print("SILVER (%s) vs. BASELINE (%s)" % (comb_mode, name_baselines[pidx]))

        assert len(silver) == len(prediction)
        assert len(silver) == config.total_num_mimic_record

        if mode == 'train':
            _evaluate(
                [silver[index] for index in config.mimic_train_indices],
                [prediction[index] for index in config.mimic_train_indices],
                func=func
            )
        elif mode == 'test':
            _evaluate(
                [silver[index] for index in config.mimic_test_indices],
                [prediction[index] for index in config.mimic_test_indices],
                func=func
            )
        elif mode == 'all':
            _evaluate(
                silver, prediction,
                func=func
            )
        elif mode == "complete":
            print("Training set")
            _evaluate(
                [silver[index] for index in config.mimic_train_indices],
                [prediction[index] for index in config.mimic_train_indices],
                func=func
            )
            print("---------")
            print("Testing set")
            _evaluate(
                [silver[index] for index in config.mimic_test_indices],
                [prediction[index] for index in config.mimic_test_indices],
                func=func
            )
            print("---------")
            print("Overall")
            _evaluate(
                silver, prediction,
                func=func
            )
        else:
            raise Exception('Invalid mode')

def evalution_of_baselines_icd(mode='test'):

    icd2fullhpo_mapping = dataloader.get_icd_hpo_silver_mapping()
    icd2limitedhpo_mapping = dataloader.get_icd_hpo_in_limited_hpo_set()
    fullhpo2icd_mapping = dataloader.get_hpo_icd_silver_mapping()
    limitedhpo2icd_mapping = dataloader.get_hpo_icd_in_limited_hpo_set()

    def filter_icd(icdstr):
        if not isinstance(icdstr, str):
            return ""
        icd_list = icdstr.split("/")
        filtered_icd_list = [icd for icd in icd_list if icd in icd2fullhpo_mapping]
        filtered_icdstr = "/".join(filtered_icd_list)
        return filtered_icdstr

    icd_code = baselines.silver_standard()['ICD9_CODE_LIST'].tolist()
    icd_code = [filter_icd(icdstr) for icdstr in icd_code]

    from itertools import combinations

    def convert_fullhpo_to_icd(hpostr):
        if not isinstance(hpostr, str):
            return ""
        hpolist = sorted(hpostr.split("/"))
        icdlist = set([icd for hpo in hpolist if hpo in fullhpo2icd_mapping for icd in fullhpo2icd_mapping[hpo]])
        # for r in range(1, len(hpolist)):
        #     for comb in combinations(hpolist, r):
        #         combstr = "/".join(comb)
        #         print(combstr)
        #         if combstr not in fullhpo2icd_mapping:
        #             continue
        #         icdlist.update(fullhpo2icd_mapping[combstr])
        # current_hposet = set(hpostr.split("/"))
        # print(current_hposet)
        # print(icd2fullhpo_mapping['4019'])
        # print(recall(icd2fullhpo_mapping['4019'], current_hposet))
        # icdlist = set()
        # for icd in icd2fullhpo_mapping:
        #     hposet = icd2fullhpo_mapping[icd]
        #     flag = True
        #     for hpo in hposet:
        #         if hpo not in current_hposet:
        #             flag = False
        #             break
        #     if flag:
        #         icdlist.add(icd)
        # print(icdlist)
        return "/".join(icdlist)

    def convert_limitedhpo_to_icd(hpostr):
        if not isinstance(hpostr, str):
            return ""
        hpolist = sorted(hpostr.split("/"))
        icdlist = set([icd for hpo in hpolist if hpo in limitedhpo2icd_mapping for icd in limitedhpo2icd_mapping[hpo]])
        # icdlist = set()
        # for r in range(1, len(hpolist)):
        #     for comb in combinations(hpolist, r):
        #         combstr = "/".join(comb)
        #         if combstr not in limitedhpo2icd_mapping:
        #             continue
        #         icdlist.update(limitedhpo2icd_mapping[combstr])
        # current_hposet = set(hpostr.split("/"))
        # print(current_hposet)
        # print(icd2limitedhpo_mapping['4019'])
        # print(recall(icd2limitedhpo_mapping['4019'], current_hposet))
        # icdlist = set()
        # for icd in icd2limitedhpo_mapping:
        #     hposet = icd2limitedhpo_mapping[icd]
        #     counter = 0
        #     for hpo in hposet:
        #         # if hpo not in current_hposet:
        #         if hpo in current_hposet:
        #             counter += 1
        #     if counter / len(hposet) > 0.49:
        #         icdlist.add(icd)
        return "/".join(icdlist)

    print("Loading baselines ...")
    # silver = baselines.silver_standard()['FULL_HPO_CODE_LIST'].tolist()
    # silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    # keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT'].tolist()
    # print("Processing baselines ...")
    # convert_fullhpo_to_icd(keyword[1])
    # print(icd_code[1])
    # exit()
    # m = [convert_limitedhpo_to_icd(hpostr) for hpostr in tqdm(silver)]
    # m = [convert_fullhpo_to_icd(hpostr) for hpostr in tqdm(silver)]
    # unsuper = decision.results_of_alpha_out(threshold=0.1, mode='all')["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"].tolist()
    # convert_limitedhpo_to_icd(unsuper[1])
    # print(icd_code[1])
    # exit()
    # m = [convert_limitedhpo_to_icd(hpostr) for hpostr in tqdm(unsuper)]
    # _evaluate(icd_code, m, func=jaccard)
    # exit()
    # _evaluate(icd_code, m, func=precision)
    # _evaluate(icd_code, m, func=recall)
    # exit()

    silver = baselines.silver_standard()['FULL_HPO_CODE_LIST'].tolist()
    keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT'].tolist()
    ncbo = baselines.ehr_phenolyzer_ncbo_annotator()["HPO_CODE_LIST_EHR_PHENO"].tolist()
    obo = baselines.obo_annotator()["HPO_CODE_LIST_OBO_ANNO"].tolist()
    # unsuper = decision.results_of_alpha_out(threshold=0.1, mode='all')["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"].tolist()

    all_baselines = [silver, keyword, ncbo, obo]
    name_baelines = ['silver', 'keyword', 'ncbo', 'obo']

    icd_baselines = list()
    for m in all_baselines:
        icd_base_list = [convert_fullhpo_to_icd(hpostr) for hpostr in tqdm(m)]
        icd_baselines.append(icd_base_list)

    # icd_unsuper = [convert_limitedhpo_to_icd(hpostr) for hpostr in tqdm(unsuper)]

    for midx, m in enumerate(icd_baselines):
        print("================")
        print(name_baelines[midx])
        _evaluate(icd_code, m, func=jaccard)
        _evaluate(icd_code, m, func=overlap_coefficient)
        _evaluate(icd_code, m, func=precision)
        _evaluate(icd_code, m, func=recall)

    # print("================")
    # print("unsuper")
    # _evaluate(icd_code, icd_unsuper, func=jaccard)
    # _evaluate(icd_code, icd_unsuper, func=overlap_coefficient)
    # _evaluate(icd_code, icd_unsuper, func=precision)
    # _evaluate(icd_code, icd_unsuper, func=recall)

def evalution_of_baselines_3digit_icd(mode='test'):

    icd2fullhpo_mapping = dataloader.get_3digit_icd_hpo_silver_mapping()
    icd2limitedhpo_mapping = dataloader.get_3digit_icd_hpo_in_limited_hpo_set()
    fullhpo2icd_mapping = dataloader.get_hpo_3digit_icd_silver_mapping()
    limitedhpo2icd_mapping = dataloader.get_hpo_3digit_icd_in_limited_hpo_set()

    def filter_icd(icdstr):
        if not isinstance(icdstr, str):
            return ""
        icd_list = icdstr.split("/")
        filtered_icd_list = [icd[:3] for icd in icd_list if icd[:3] in icd2fullhpo_mapping]
        filtered_icdstr = "/".join(filtered_icd_list)
        return filtered_icdstr

    icd_code = baselines.silver_standard()['ICD9_CODE_LIST'].tolist()

    # icd_code_in_mimic_set = set([icd for icdstr in icd_code for icd in icdstr.split("/") if len(icd) > 0])
    # print(len(icd_code_in_mimic_set)) # 6918
    # exit()

    icd_code = [filter_icd(icdstr) for icdstr in icd_code]

    # icd_3digit__code_in_mimic_set = set([icd for icdstr in icd_code for icd in icdstr.split("/") if len(icd) > 0])
    # print(icd_3digit_code_in_mimic_set)
    # print(len(icd_3digit_code_in_mimic_set)) # 66
    # exit()

    from itertools import combinations

    def convert_fullhpo_to_icd(hpostr):
        if not isinstance(hpostr, str):
            return ""
        hpolist = sorted(hpostr.split("/"))
        icdlist = set([icd for hpo in hpolist if hpo in fullhpo2icd_mapping for icd in fullhpo2icd_mapping[hpo]])
        # for r in range(1, len(hpolist)):
        #     for comb in combinations(hpolist, r):
        #         combstr = "/".join(comb)
        #         print(combstr)
        #         if combstr not in fullhpo2icd_mapping:
        #             continue
        #         icdlist.update(fullhpo2icd_mapping[combstr])
        # current_hposet = set(hpostr.split("/"))
        # print(current_hposet)
        # print(icd2fullhpo_mapping['4019'])
        # print(recall(icd2fullhpo_mapping['4019'], current_hposet))
        # icdlist = set()
        # for icd in icd2fullhpo_mapping:
        #     hposet = icd2fullhpo_mapping[icd]
        #     flag = True
        #     for hpo in hposet:
        #         if hpo not in current_hposet:
        #             flag = False
        #             break
        #     if flag:
        #         icdlist.add(icd)
        # print(icdlist)
        return "/".join(icdlist)

    def convert_limitedhpo_to_icd(hpostr):
        if not isinstance(hpostr, str):
            return ""
        hpolist = sorted(hpostr.split("/"))
        icdlist = set([icd for hpo in hpolist if hpo in limitedhpo2icd_mapping for icd in limitedhpo2icd_mapping[hpo]])
        # icdlist = set()
        # for r in range(1, len(hpolist)):
        #     for comb in combinations(hpolist, r):
        #         combstr = "/".join(comb)
        #         if combstr not in limitedhpo2icd_mapping:
        #             continue
        #         icdlist.update(limitedhpo2icd_mapping[combstr])
        # current_hposet = set(hpostr.split("/"))
        # print(current_hposet)
        # print(icd2limitedhpo_mapping['4019'])
        # print(recall(icd2limitedhpo_mapping['4019'], current_hposet))
        # icdlist = set()
        # for icd in icd2limitedhpo_mapping:
        #     hposet = icd2limitedhpo_mapping[icd]
        #     counter = 0
        #     for hpo in hposet:
        #         # if hpo not in current_hposet:
        #         if hpo in current_hposet:
        #             counter += 1
        #     if counter / len(hposet) > 0.49:
        #         icdlist.add(icd)
        return "/".join(icdlist)

    print("Loading baselines ...")
    # silver = baselines.silver_standard()['FULL_HPO_CODE_LIST'].tolist()
    # silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    # keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT'].tolist()
    # print("Processing baselines ...")
    # convert_fullhpo_to_icd(keyword[1])
    # print(icd_code[1])
    # exit()
    # m = [convert_limitedhpo_to_icd(hpostr) for hpostr in tqdm(silver)]
    # m = [convert_fullhpo_to_icd(hpostr) for hpostr in tqdm(silver)]
    # unsuper = decision.results_of_alpha_out(threshold=0.1, mode='all')["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"].tolist()
    # convert_limitedhpo_to_icd(unsuper[1])
    # print(icd_code[1])
    # exit()
    # m = [convert_limitedhpo_to_icd(hpostr) for hpostr in tqdm(unsuper)]
    # _evaluate(icd_code, m, func=jaccard)
    # exit()
    # _evaluate(icd_code, m, func=precision)
    # _evaluate(icd_code, m, func=recall)
    # exit()

    # silver = baselines.silver_standard()['FULL_HPO_CODE_LIST'].tolist()

    silver = []
    for icdstr in icd_code:
        icdlist = icdstr.split("/")
        hposet = set()
        for icd in icdlist:
            if len(icd) == 0:
                continue
            hposet.update(icd2fullhpo_mapping[icd])
        silver.append("/".join(hposet))

    hpo_to_hpoidx = {hpo: hidx for hidx, hpo in enumerate(dataloader.hpo_limited_list)}
    limited_hpo_counter = [0] * len(dataloader.hpo_limited_list)
    for icdstr in icd_code:
        icdlist = icdstr.split("/")
        hposet = set()
        for icd in icdlist:
            if len(icd) == 0:
                continue
            hposet.update(icd2limitedhpo_mapping[icd])
        for hpo in hposet:
            limited_hpo_counter[hpo_to_hpoidx[hpo]] += 1

    for hidx in range(len(dataloader.hpo_limited_list)):
        print(hidx, dataloader.hpo_limited_list[hidx], limited_hpo_counter[hidx], len(icd_code))

    keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT'].tolist()
    ncbo = baselines.ehr_phenolyzer_ncbo_annotator()["HPO_CODE_LIST_EHR_PHENO"].tolist()
    obo = baselines.obo_annotator()["HPO_CODE_LIST_OBO_ANNO"].tolist()
    metamap = baselines.aggregate_metamap_results()["HPO_CODE_LIST_METAMAP_ANNO_PREDECESSORS_ONLY"].tolist()
    # unsuper = decision.results_of_alpha_out(threshold=0.1, mode='all')["HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY"].tolist()
    import playground
    unsuper = playground.analyze_appearance_of_hpo(12)

    all_baselines = [silver, keyword, ncbo, obo, metamap]
    name_baelines = ['silver', 'keyword', 'ncbo', 'obo', 'metamap']

    icd_baselines = list()
    for m in all_baselines:
        icd_base_list = [convert_fullhpo_to_icd(hpostr) for hpostr in tqdm(m)]
        icd_baselines.append(icd_base_list)

    icd_unsuper = [convert_limitedhpo_to_icd(hpostr) for hpostr in tqdm(unsuper)]

    for midx, m in enumerate(icd_baselines):
        print("================")
        print(name_baelines[midx])
        _evaluate(icd_code, m, func=jaccard)
        _evaluate(icd_code, m, func=overlap_coefficient)
        _evaluate(icd_code, m, func=precision)
        _evaluate(icd_code, m, func=recall)

    print("================")
    print("unsuper")
    _evaluate(icd_code, icd_unsuper, func=jaccard)
    _evaluate(icd_code, icd_unsuper, func=overlap_coefficient)
    _evaluate(icd_code, icd_unsuper, func=precision)
    _evaluate(icd_code, icd_unsuper, func=recall)

def evaluation_of_random_pick(avg_hpo, func, mode='test'):

    silver = get_icd2hpo_3digit_results()
    prediction = baselines.random_pick(avg_hpo)

    _evaluate_with_mode(silver, prediction, func, mode)

def _evaluate_with_mode(silver, prediction, func, mode):

    if mode == 'train':
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [prediction[index] for index in config.mimic_train_indices],
            func=func
        )
    elif mode == 'test':
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [prediction[index] for index in config.mimic_test_indices],
            func=func
        )
    elif mode == 'all':
        _evaluate(
            silver, prediction,
            func=func
        )
    elif mode == "complete":
        print("Training set")
        _evaluate(
            [silver[index] for index in config.mimic_train_indices],
            [prediction[index] for index in config.mimic_train_indices],
            func=func
        )
        print("---------")
        print("Testing set")
        _evaluate(
            [silver[index] for index in config.mimic_test_indices],
            [prediction[index] for index in config.mimic_test_indices],
            func=func
        )
        print("---------")
        print("Overall")
        _evaluate(
            silver, prediction,
            func=func
        )
    else:
        raise Exception('Invalid mode')

def evaluation_between_all_methods(func):
    silver = get_icd2hpo_3digit_results()
    keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY'].tolist()
    ncbo = baselines.ehr_phenolyzer_ncbo_annotator()["HPO_CODE_LIST_EHR_PHENO_PREDECESSORS_ONLY"].tolist()
    obo = baselines.obo_annotator()["HPO_CODE_LIST_OBO_ANNO_PREDECESSORS_ONLY"].tolist()
    metamap = baselines.aggregate_metamap_results()["HPO_CODE_LIST_METAMAP_ANNO_PREDECESSORS_ONLY"].tolist()
    unsuper = decision.annotation_with_threshold()
    all_methods = [silver, keyword, ncbo, obo, metamap, unsuper]
    name_methods = ['silver', 'keyword', 'ncbo', 'obo', 'metamap', 'unsuper']

    print(func.__name__)
    for i in range(len(all_methods)):
        for j in range(len(all_methods)):
            print("==========")
            print("%s vs. %s" % (name_methods[i], name_methods[j]))
            _evaluate_with_mode(all_methods[i], all_methods[j], func, mode='test')

def evaluation_unsuper_case_study():
    silver = get_icd2hpo_3digit_results()
    unsuper = decision.annotation_with_threshold()

    _evaluate_with_mode(silver, unsuper, jaccard, mode='all')

def get_icd_code_3digit():
    icd2limitedhpo_mapping = dataloader.get_3digit_icd_hpo_in_limited_hpo_set()

    def filter_icd(icdstr):
        if not isinstance(icdstr, str):
            return ""
        icd_list = icdstr.split("/")
        filtered_icd_list = [icd[:3] for icd in icd_list if icd[:3] in icd2limitedhpo_mapping]
        filtered_icdstr = "/".join(filtered_icd_list)
        return filtered_icdstr

    icd_code = baselines.silver_standard()['ICD9_CODE_LIST'].tolist()

    icd_code = [filter_icd(icdstr) for icdstr in icd_code]
    return icd_code

def evaluate_for_each_icd():
    # silver = get_icd2hpo_3digit_results()
    unsuper = decision.annotation_with_threshold()

    icd_code = get_icd_code_3digit()

    icd2limitedhpo_mapping = dataloader.get_3digit_icd_hpo_in_limited_hpo_set()

    def _check_icd(icd, icdstr):
        if not isinstance(icdstr, str) or len(icdstr) == 0:
            return False
        icdlist = icdstr.split("/")
        return icd in icdlist

    results = dict()
    for icd in tqdm(icd2limitedhpo_mapping):
        new_silver = []
        new_unsuper = []
        for idx, icdstr in enumerate(icd_code):
            if _check_icd(icd, icdstr):
                new_silver.append("/".join(icd2limitedhpo_mapping[icd]))
                new_unsuper.append(unsuper[idx])
            else:
                new_silver.append("")
                new_unsuper.append("")
        _, r = _evaluate(
            [new_silver[index] for index in config.mimic_test_indices],
            [new_unsuper[index] for index in config.mimic_test_indices],
            func=recall
        )
        results[icd] = r

    sorted_results = sorted(results.items(), key=lambda kv: kv[1])

    print("worst")
    for i in range(10):
        print(sorted_results[i])
        print(icd2limitedhpo_mapping[sorted_results[i][0]])
    print("=========")
    print("best")
    for i in range(10):
        print(sorted_results[-i-1])
        print(icd2limitedhpo_mapping[sorted_results[-i-1][0]])



if __name__ == '__main__':

    '''
    evaluate_keyword_search(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY',
        func=overlap_coefficient,
        mode='complete'
    )
    '''
    '''
    evaluate_keyword_search(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY',
        func=jaccard,
        mode='all'
    )
    '''
    '''
    evaluate_keyword_search_with_negation(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY_WITH_NEGATION',
        func=overlap_coefficient,
        mode='test'
    )
    '''
    '''
    evaluate_keyword_search_with_negation(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY_WITH_NEGATION',
        func=jaccard,
        mode='test'
    )
    '''
    '''
    evaluate_unsupervised_method(
        column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY",
        threshold=0.5,
        decision_mode='argmax',
        func=jaccard,
        mode="complete"
    )
    '''
    '''
    evaluate_unsupervised_method(
        column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY",
        threshold=0.5,
        decision_mode='argmax',
        func=overlap_coefficient,
        mode="complete"
    )
    '''
    '''
    evaluate_unsupervised_method(
        column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY",
        threshold=0.2,
        decision_mode='all',
        func=overlap_coefficient,
        mode="complete"
    )
    '''
    # evaluation_between_all_methods(jaccard)
    # print("====================")
    # evaluation_between_all_methods(overlap_coefficient)
    # _global_print_best_worst_case = True
    # evaluation_unsuper_case_study()
    # exit()

    # icd_code = get_icd_code_3digit()
    # print(icd_code[39934])
    # print(icd_code[26491])
    # print(icd_code[8821])

    # mimic_data_text = dataloader.load_mimic()["TEXT"].tolist()
    # print(mimic_data_text[42292])

    evaluate_for_each_icd()

    pass





