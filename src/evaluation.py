import numpy as np

# reference: https://en.wikipedia.org/wiki/Overlap_coefficient
def overlap_coefficient(list_of_set1, list_of_set2):

    overlap_list = list()
    at_least_one_match_list = list()

    assert len(list_of_set1) == len(list_of_set2)
    for i in range(len(list_of_set1)):
        if isinstance(list_of_set1[i], float) or isinstance(list_of_set2[i], float):
            continue
        hposet1 = set(list_of_set1[i].split("/"))
        hposet2 = set(list_of_set2[i].split("/"))
        if len(hposet1) > 0 and len(hposet2) > 0:
            overlap = len(hposet1 & hposet2) / min(len(hposet1), len(hposet2))
            at_least_one_match = 1 if len(hposet1 & hposet2) > 0 else 0
            overlap_list.append(overlap)
            at_least_one_match_list.append(at_least_one_match)

    print("Total num of records: %d" % len(list_of_set1))
    print("Total num of valid comparison: %d" % len(overlap_list))
    print("Avg overlapping coefficient: %.8f" % np.mean(overlap_list))
    print("Median overlapping coefficient: %.8f" % np.median(overlap_list))
    print("Avg at least one matched: %.8f" % np.mean(at_least_one_match_list))
    print("Median at least one matched: %.8f" % np.median(at_least_one_match_list))

# reference: https://en.wikipedia.org/wiki/Jaccard_index
def jaccard(list_of_set1, list_of_set2):

    overlap_list = list()
    at_least_one_match_list = list()

    assert len(list_of_set1) == len(list_of_set2)

    score_dict = dict()
    for i in range(len(list_of_set1)):
        if isinstance(list_of_set1[i], float) or isinstance(list_of_set2[i], float):
            continue
        hposet1 = set(list_of_set1[i].split("/"))
        hposet2 = set(list_of_set2[i].split("/"))
        if len(hposet1) > 0 and len(hposet2) > 0:
            overlap = len(hposet1 & hposet2) / len(hposet1 | hposet2)
            at_least_one_match = 1 if len(hposet1 & hposet2) > 0 else 0
            overlap_list.append(overlap)
            at_least_one_match_list.append(at_least_one_match)

            score_dict[i] = overlap

    '''
    sorted_score = sorted(score_dict.items(), key=lambda kv: kv[1])

    # worst case
    for i in range(10):
        print(i, sorted_score[i])
        print(list_of_set1[sorted_score[i][0]].split("/"))
        print(list_of_set2[sorted_score[i][0]].split("/"))
        print("---")

    print("====")

    # best case
    for i in range(1, 11):
        print(-i, sorted_score[-i])
        print(list_of_set1[sorted_score[-i][0]].split("/"))
        print(list_of_set2[sorted_score[-i][0]].split("/"))
        print("---")
    '''

    print("Total num of records: %d" % len(list_of_set1))
    print("Total num of valid comparison: %d" % len(overlap_list))
    print("Avg jaccard index: %.8f" % np.mean(overlap_list))
    print("Median jaccard index: %.8f" % np.median(overlap_list))
    print("Avg at least one matched: %.8f" % np.mean(at_least_one_match_list))
    print("Median at least one matched: %.8f" % np.median(at_least_one_match_list))


def evaluate_keyword_search_all_hpo_with_overlapping_cofficient():
    import config
    import baselines

    silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT'].tolist()
    assert len(silver) == len(keyword)
    assert len(silver) == config.total_num_mimic_record

    overlap_coefficient(
        [silver[index] for index in config.mimic_train_indices],
        [keyword[index] for index in config.mimic_train_indices],
    )

    print("-----")

    overlap_coefficient(
        [silver[index] for index in config.mimic_test_indices],
        [keyword[index] for index in config.mimic_test_indices],
    )

    print("-----")

    overlap_coefficient(
        silver, keyword
    )

def evaluate_keyword_search_subset_hpo_with_jaccard():
    import config
    import baselines

    silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    keyword = baselines.keyword_search()['HPO_CODE_LIST_KEYWORD_SEARCH_WITHOUT_PARENT'].tolist()
    assert len(silver) == len(keyword)
    assert len(silver) == config.total_num_mimic_record

    subset_hpo = set()
    for hpostr in silver:
        if isinstance(hpostr, float):
            continue
        hpolist = hpostr.split("/")
        for hpo in hpolist:
            subset_hpo.add(hpo)

    filtered_keyword = list()
    for k in keyword:
        if isinstance(k, float):
            filtered_keyword.append(k)
        else:
            hpolist = k.split("/")
            filtered_hpolist = [hpo for hpo in hpolist if hpo in subset_hpo]
            filtered_hpostr = "/".join(filtered_hpolist)
            filtered_keyword.append(filtered_hpostr)

    jaccard(
        [silver[index] for index in config.mimic_train_indices],
        [filtered_keyword[index] for index in config.mimic_train_indices],
    )

    print("-----")

    jaccard(
        [silver[index] for index in config.mimic_test_indices],
        [filtered_keyword[index] for index in config.mimic_test_indices],
    )

    print("-----")

    jaccard(
        silver, filtered_keyword
    )

if __name__ == '__main__':
    evaluate_keyword_search_all_hpo_with_overlapping_cofficient()
    # WITHOUT propagation to parent HPO ndoes
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg overlapping coefficient: 0.05439286
    # Median overlapping coefficient: 0.06250000
    # Avg at least one matched: 0.63379419
    # Median at least one matched: 1.00000000
    # -----
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg overlapping coefficient: 0.05522266
    # Median overlapping coefficient: 0.06896552
    # Avg at least one matched: 0.64215803
    # Median at least one matched: 1.00000000
    # -----
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27979
    # Avg overlapping coefficient: 0.05464134
    # Median overlapping coefficient: 0.06250000
    # Avg at least one matched: 0.63629865
    # Median at least one matched: 1.00000000

    print("======")

    evaluate_keyword_search_subset_hpo_with_jaccard()
    # WITHOUT propagation to parent HPO nodes
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.02606812
    # Median jaccard index: 0.02941176
    # Avg at least one matched: 0.63379419
    # Median at least one matched: 1.00000000
    # -----
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.02655940
    # Median jaccard index: 0.03030303
    # Avg at least one matched: 0.64215803
    # Median at least one matched: 1.00000000
    # -----
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27979
    # Avg jaccard index: 0.02621523
    # Median jaccard index: 0.02941176
    # Avg at least one matched: 0.63629865
    # Median at least one matched: 1.00000000



