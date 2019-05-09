import numpy as np
import dataloader

_set_of_hpos = set(dataloader.hpo_limited_list)

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

        for hpo in (hposet1 | hposet2):
            assert hpo in _set_of_hpos

        if len(hposet1) > 0 and len(hposet2) > 0:
            overlap = func(hposet1, hposet2)
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
    print("Avg %s index: %.8f" % (func.__name__, np.mean(overlap_list)))
    print("Median %s index: %.8f" % (func.__name__, np.median(overlap_list)))
    print("Avg at least one matched: %.8f" % np.mean(at_least_one_match_list))
    print("Median at least one matched: %.8f" % np.median(at_least_one_match_list))

    return score_dict


def evaluate_keyword_search(column_of_keyword, func, mode='all'):
    import config
    import baselines

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
    import config
    import baselines

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

def evaluate_unsupervised_method(column_of_keyword, threshold, decision_mode, func, mode='complete'):
    import config
    import baselines
    import decision

    silver = baselines.silver_standard()['HPO_CODE_LIST'].tolist()
    unsuper = decision.results_of_alpha_out(threshold=threshold, mode=decision_mode)[column_of_keyword].tolist()
    assert len(silver) == len(unsuper)
    assert len(silver) == config.total_num_mimic_record

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

if __name__ == '__main__':

    '''
    evaluate_keyword_search(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY',
        func=overlap_coefficient,
        mode='complete'
    )
    '''
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19585
    # Avg overlap_coefficient index: 0.55367395
    # Median overlap_coefficient index: 0.60000000
    # Avg at least one matched: 0.99397498
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8373
    # Avg overlap_coefficient index: 0.55553882
    # Median overlap_coefficient index: 0.60000000
    # Avg at least one matched: 0.99570047
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27958
    # Avg overlap_coefficient index: 0.55423245
    # Median overlap_coefficient index: 0.60000000
    # Avg at least one matched: 0.99449174
    # Median at least one matched: 1.00000000

    evaluate_keyword_search(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY',
        func=jaccard,
        mode='test'
    )
    # part of mimic
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19585
    # Avg jaccard index: 0.27036505
    # Median jaccard index: 0.25000000
    # Avg at least one matched: 0.99397498
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8373
    # Avg jaccard index: 0.27225081
    # Median jaccard index: 0.25000000
    # Avg at least one matched: 0.99570047
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27958
    # Avg jaccard index: 0.27092981
    # Median jaccard index: 0.25000000
    # Avg at least one matched: 0.99449174
    # Median at least one matched: 1.00000000
    # ======================================================
    # whole mimic
    # WITH DIRECT CHILDREN OF HP:0000118 ONLY
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.28361271
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99836743
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.28633937
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99880640
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27979
    # Avg jaccard index: 0.28442918
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99849887
    # Median at least one matched: 1.00000000

    '''
    evaluate_keyword_search_with_negation(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY_WITH_NEGATION',
        func=overlap_coefficient,
        mode='test'
    )
    '''
    # WITH DIRECT CHILDREN OF HP:0000118 ONLY
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg overlap_coefficient index: 0.63943633
    # Median overlap_coefficient index: 0.60000000
    # Avg at least one matched: 0.99836743
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg overlap_coefficient index: 0.64104041
    # Median overlap_coefficient index: 0.60000000
    # Avg at least one matched: 0.99868704
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27979
    # Avg overlap_coefficient index: 0.63991665
    # Median overlap_coefficient index: 0.60000000
    # Avg at least one matched: 0.99846313
    # Median at least one matched: 1.00000000

    '''
    evaluate_keyword_search_with_negation(
        column_of_keyword='HPO_CODE_LIST_KEYWORD_SEARCH_PREDECESSORS_ONLY_WITH_NEGATION',
        func=jaccard,
        mode='test'
    )
    '''
    # WITH DIRECT CHILDREN OF HP:0000118 ONLY
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.28489161
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99836743
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.28738001
    # Median jaccard index: 0.27272727
    # Avg at least one matched: 0.99868704
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27979
    # Avg jaccard index: 0.28563673
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99846313
    # Median at least one matched: 1.00000000

    '''
    evaluate_unsupervised_method(
        column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY",
        threshold=0.5,
        decision_mode='argmax',
        func=jaccard,
        mode="complete"
    )
    '''
    #############################
    # argmax
    # threshold 0.5
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19418
    # Avg jaccard index: 0.21260561
    # Median jaccard index: 0.20000000
    # Avg at least one matched: 0.90622103
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8303
    # Avg jaccard index: 0.21431416
    # Median jaccard index: 0.20000000
    # Avg at least one matched: 0.91099603
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27721
    # Avg jaccard index: 0.21311736
    # Median jaccard index: 0.20000000
    # Avg at least one matched: 0.90765124
    # Median at least one matched: 1.00000000
    # ===========
    # threshold 0.4
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.24501064
    # Median jaccard index: 0.25000000
    # Avg at least one matched: 0.96122647
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.24701272
    # Median jaccard index: 0.25000000
    # Avg at least one matched: 0.96216281
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27980
    # Avg jaccard index: 0.24560136
    # Median jaccard index: 0.25000000
    # Avg at least one matched: 0.96147248
    # Median at least one matched: 1.00000000
    # ===========
    # threshold 0.3
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.27176006
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.98898015
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.27326334
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.98997374
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27980
    # Avg jaccard index: 0.27220047
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.98924232
    # Median at least one matched: 1.00000000
    # ==========
    # threshold 0.2
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.28264217
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99688791
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.28625404
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99773216
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27980
    # Avg jaccard index: 0.28371356
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99710508
    # Median at least one matched: 1.00000000
    # =============
    # threshold 0.1
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.28237558
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.99826539
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.28590053
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.99856768
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27980
    # Avg jaccard index: 0.28342096
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.99832023
    # Median at least one matched: 1.00000000
    # ============
    # threshold 0.0
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.28236946
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.99826539
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.28591972
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.99856768
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27980
    # Avg jaccard index: 0.28342242
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.99832023
    # Median at least one matched: 1.00000000

    '''
    evaluate_unsupervised_method(
        column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY",
        threshold=0.5,
        decision_mode='argmax',
        func=overlap_coefficient,
        mode="complete"
    )
    '''
    # threshold 0.2
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg overlap_coefficient index: 0.72547395
    # Median overlap_coefficient index: 0.77777778
    # Avg at least one matched: 0.99688791
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg overlap_coefficient index: 0.73043624
    # Median overlap_coefficient index: 0.80000000
    # Avg at least one matched: 0.99773216
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27979
    # Avg overlap_coefficient index: 0.72695985
    # Median overlap_coefficient index: 0.78571429
    # Avg at least one matched: 0.99714071
    # Median at least one matched: 1.00000000

    pass

    '''
    evaluate_unsupervised_method(
        column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY",
        threshold=0.2,
        decision_mode='all',
        func=jaccard,
        mode="complete"
    )
    '''

    #############################
    # all higher than threshold
    # threshold 0.2
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg jaccard index: 0.28555203
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.99801031
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg jaccard index: 0.28877882
    # Median jaccard index: 0.26666667
    # Avg at least one matched: 0.99868704
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27980
    # Avg jaccard index: 0.28650801
    # Median jaccard index: 0.26315789
    # Avg at least one matched: 0.99817727
    # Median at least one matched: 1.00000000

    '''
    evaluate_unsupervised_method(
        column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY",
        threshold=0.2,
        decision_mode='all',
        func=overlap_coefficient,
        mode="complete"
    )
    '''
    #############################
    # all higher than threshold
    # threshold 0.2
    # Training set
    # Total num of records: 36905
    # Total num of valid comparison: 19601
    # Avg overlap_coefficient index: 0.78489369
    # Median overlap_coefficient index: 0.80000000
    # Avg at least one matched: 0.99801031
    # Median at least one matched: 1.00000000
    # ---------
    # Testing set
    # Total num of records: 15816
    # Total num of valid comparison: 8378
    # Avg overlap_coefficient index: 0.78972590
    # Median overlap_coefficient index: 0.80000000
    # Avg at least one matched: 0.99868704
    # Median at least one matched: 1.00000000
    # ---------
    # Overall
    # Total num of records: 52722
    # Total num of valid comparison: 27979
    # Avg overlap_coefficient index: 0.78634064
    # Median overlap_coefficient index: 0.80000000
    # Avg at least one matched: 0.99821295
    # Median at least one matched: 1.00000000

    pass

    '''
    evaluate_unsupervised_method(
        column_of_keyword="HPO_CODE_LIST_UNSUPERVISED_METHOD_PREDECESSORS_ONLY",
        threshold=0.5,
        decision_mode='all',
        func=jaccard,
        mode="complete"
    )
    '''




