import numpy as np

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
        hposet1 = set(list_of_set1[i].split("/"))
        hposet2 = set(list_of_set2[i].split("/"))
        if len(hposet1) > 0 and len(hposet2) > 0:
            overlap = func(hposet1, hposet2)
            at_least_one_match = 1 if len(hposet1 & hposet2) > 0 else 0
            overlap_list.append(overlap)
            at_least_one_match_list.append(at_least_one_match)

            score_dict[i] = overlap

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

    print("Total num of records: %d" % len(list_of_set1))
    print("Total num of valid comparison: %d" % len(overlap_list))
    print("Avg %s index: %.8f" % (func.__name__, np.mean(overlap_list)))
    print("Median %s index: %.8f" % (func.__name__, np.median(overlap_list)))
    print("Avg at least one matched: %.8f" % np.mean(at_least_one_match_list))
    print("Median at least one matched: %.8f" % np.median(at_least_one_match_list))


def evaluate_keyword_search(column_of_keyword, func, mode='all'):
    import config
    import baselines

    silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    keyword = baselines.keyword_search()[column_of_keyword].tolist()
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

if __name__ == '__main__':

    '''
    evaluate_keyword_search(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY',
        func=overlap_coefficient,
        mode='test'
    )
    '''
    # WITH DIRECT CHILDREN OF HP:0000118 ONLY
    # Training set
    # Total num of records: 36905                                                                                           │·····················································································
    # Total num of valid comparison: 19601                                                                                  │·····················································································
    # Avg overlap_coefficient index: 0.65287205                                                                             │·····················································································
    # Median overlap_coefficient index: 0.60000000                                                                          │·····················································································
    # Avg at least one matched: 0.99836743                                                                                  │·····················································································
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816                                                                                           │·····················································································
    # Total num of valid comparison: 8378                                                                                   │·····················································································
    # Avg overlap_coefficient index: 0.65531696                                                                             │·····················································································
    # Median overlap_coefficient index: 0.60000000                                                                          │·····················································································
    # Avg at least one matched: 0.99880640                                                                                  │·····················································································
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722                                                                                           │·····················································································
    # Total num of valid comparison: 27979                                                                                  │·····················································································
    # Avg overlap_coefficient index: 0.65360415                                                                             │·····················································································
    # Median overlap_coefficient index: 0.60000000                                                                          │·····················································································
    # Avg at least one matched: 0.99849887                                                                                  │·····················································································
    # Median at least one matched: 1.00000000


    '''
    evaluate_keyword_search(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY',
        func=jaccard,
        mode='test'
    )
    '''
    # WITH DIRECT CHILDREN OF HP:0000118 ONLY
    # Training set
    # Total num of records: 36905                                                                                           │·····················································································
    # Total num of valid comparison: 19601                                                                                  │·····················································································
    # Avg jaccard index: 0.28328037                                                                                         │·····················································································
    # Median jaccard index: 0.26666667                                                                                      │·····················································································
    # Avg at least one matched: 0.99836743                                                                                  │·····················································································
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816                                                                                           │·····················································································
    # Total num of valid comparison: 8378                                                                                   │·····················································································
    # Avg jaccard index: 0.28600811                                                                                         │·····················································································
    # Median jaccard index: 0.26666667                                                                                      │·····················································································
    # Avg at least one matched: 0.99880640                                                                                  │·····················································································
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722                                                                                           │·····················································································
    # Total num of valid comparison: 27979                                                                                  │·····················································································
    # Avg jaccard index: 0.28409716                                                                                         │·····················································································
    # Median jaccard index: 0.26666667                                                                                      │·····················································································
    # Avg at least one matched: 0.99849887                                                                                  │·····················································································
    # Median at least one matched: 1.00000000



