# Medical Text

### To Kai and Chengliang

* analyze ICD->HPO vs. Keyword Search
* run `python evaluation.py` to get worst / best cases, e.g., in printout
    - `0 (385, 0.0)`
    - `['HP:0025031', 'HP:0000119', 'HP:0001939', 'HP:0002086']`
    - `['HP:0001871', 'HP:0001626', 'HP:0025142', 'HP:0000707', 'HP:0002715']`
    - This is an example of worst case. `385` is the EHR id. `0.0` is the Jaccard.
    - The first line is the results of ICD->HPO. (unsorted)
    - The second line is the results of keyword search. (unsorted)
    - Check code `evaluate_keyword_search()` in `src/evaluation.py` for details.
    - Files:
        - `outputs/results/silver_standard.csv` for ICD->HPO results
        - `outputs/results/keyword_search.csv` for keyword search results
        - `data/MIMIC/ready.csv` for all EHRs (MIMIC) (both original and cleaned) 
* run `python playground.py` to analyze a specific EHR e.g. the EHR with id `385`.
    - You may change the EHR id in at the bottom of `playground.py`.
    - Check code `analyze_specific_case()` in  `src/playground.py` for details.
* To get ICD->HPO mapping
    - ICD -> OMIM
        - File: `data/HPO/icd2omim.json`
        - Code: `get_icd_omim_mapping()` in `src/dataloader.py`
    - OMIM -> Full HPO (all HPO terms)
        - File: `data/HPO/omim2hpo.json`
        - Code: `get_omim_hpo_mapping()` in `src/dataloader.py`
    - ICD -> OMIM -> Full HPO
        - File: `data/HPO/icd2fullhpo.json`
        - Code: `get_icd_hpo_silver_mapping()` in `src/dataloader.py`
    - ICD -> OMIM -> Limited HPO (24 phenotypic abnormalities only)
        - File: `data/HPO/icd2limitedhpo.json`
        - Code: `get_icd_hpo_in_limited_hpo_set()` in `src/dataloader.py`

### Experiments

##### Evaluation on HPO

###### 3-digit ICD 2 HPO

- Training set
- SILVER: 3-digit ICD to HPO mapping
- SILVER vs. BASELINE (keyword)
    - Avg jaccard index: 0.36742858
    - Median jaccard index: 0.36842105
    - Avg overlap_coefficient index: 0.75779701
    - Median overlap_coefficient index: 0.81818182
- SILVER vs. BASELINE (ncbo)
    - Avg jaccard index: 0.41346371
    - Median jaccard index: 0.42105263
    - Avg overlap_coefficient index: 0.78851977
    - Median overlap_coefficient index: 0.83333333
- SILVER vs. BASELINE (obo)
    - Avg jaccard index: 0.42542502
    - Median jaccard index: 0.42857143
    - Avg overlap_coefficient index: 0.79855777
    - Median overlap_coefficient index: 0.84615385
- SILVER vs. BASELINE (metamap)
    - Avg jaccard index: 0.40360925
    - Median jaccard index: 0.41666667
    - Avg overlap_coefficient index: 0.82337229
    - Median overlap_coefficient index: 0.90000000
- SILVER vs. OURS (human curated threshold)
    - Avg jaccard index: 0.49020323
    - Median jaccard index: 0.50000000
    - Avg overlap_coefficient index: 0.85105482
    - Median overlap_coefficient index: 0.88888889
- SILVER vs. OURS (normalized top 20%)
    - Avg jaccard index: 0.48786125
    - Median jaccard index: 0.50000000
    - Avg overlap_coefficient index: 0.84513326
    - Median overlap_coefficient index: 0.87500000

- Testing set
- SILVER: 3-digit ICD to HPO mapping
- SILVER vs. BASELINE (keyword)
    - Avg jaccard index: 0.36702556
    - Median jaccard index: 0.36842105
    - Avg overlap_coefficient index: 0.75490689
    - Median overlap_coefficient index: 0.80000000
- SILVER vs. BASELINE (ncbo)
    - Avg jaccard index: 0.41441987
    - Median jaccard index: 0.42105263
    - Avg overlap_coefficient index: 0.78626944
    - Median overlap_coefficient index: 0.83333333
- SILVER vs. BASELINE (obo)
    - Avg jaccard index: 0.42672483
    - Median jaccard index: 0.43478261
    - Avg overlap_coefficient index: 0.79674434
    - Median overlap_coefficient index: 0.84615385
- SILVER vs. BASELINE (metamap)
    - Avg jaccard index: 0.40379370
    - Median jaccard index: 0.41666667
    - Avg overlap_coefficient index: 0.82268007
    - Median overlap_coefficient index: 0.90000000
- SILVER vs. OURS (human curated threshold)
    - Avg jaccard index: 0.49432106
    - Median jaccard index: 0.50000000
    - Avg overlap_coefficient index: 0.85149958
    - Median overlap_coefficient index: 0.88888889 
- SILVER vs. OURS (normalized top 20%)
    - Avg jaccard index: 0.48957082
    - Median jaccard index: 0.50000000
    - Avg overlap_coefficient index: 0.84550405
    - Median overlap_coefficient index: 0.87500000


Statistics
- Keyword
    - Avg HPO for all 9
    - Median HPO for all 9
    - Avg HPO for those have 9
    - Median HPO for those have 9
- NCBO
    - Avg HPO for all 10
    - Median HPO for all 10
    - Avg HPO for those have 10
    - Median HPO for those have 10
- OBO
    - Avg HPO for all 11
    - Median HPO for all 11
    - Avg HPO for those have 11
    - Median HPO for those have 11
- MetaMap
    - Avg HPO for all 8
    - Median HPO for all 9
    - Avg HPO for those have 8
    - Median HPO for those have 9

- Avg of HPO / EHR: 1 
    - Avg jaccard index: 0.04167951
    - Median jaccard index: 0.04761905
    - Avg overlap_coefficient index: 0.55038226
    - Median overlap_coefficient index: 1.00000000
- Avg of HPO / EHR: 2 
    - Avg jaccard index: 0.07766547
    - Median jaccard index: 0.08333333
    - Avg overlap_coefficient index: 0.55424312
    - Median overlap_coefficient index: 0.50000000
- Avg of HPO / EHR: 3 
    - Avg jaccard index: 0.11045807
    - Median jaccard index: 0.11111111
    - Avg overlap_coefficient index: 0.55889399
    - Median overlap_coefficient index: 0.66666667
- Avg of HPO / EHR: 4 
    - Avg jaccard index: 0.14219003
    - Median jaccard index: 0.15000000
    - Avg overlap_coefficient index: 0.55695719
    - Median overlap_coefficient index: 0.50000000
- Avg of HPO / EHR: 5 
    - Avg jaccard index: 0.17238019
    - Median jaccard index: 0.18750000
    - Avg overlap_coefficient index: 0.56083461
    - Median overlap_coefficient index: 0.60000000
- Avg of HPO / EHR: 6 
    - Avg jaccard index: 0.19866960
    - Median jaccard index: 0.21428571
    - Avg overlap_coefficient index: 0.57049567
    - Median overlap_coefficient index: 0.66666667
- Avg of HPO / EHR: 7 
    - Avg jaccard index: 0.22573697
    - Median jaccard index: 0.23809524
    - Avg overlap_coefficient index: 0.57827126
    - Median overlap_coefficient index: 0.57142857
- Avg of HPO / EHR: 8 
    - Avg jaccard index: 0.25104061
    - Median jaccard index: 0.27272727
    - Avg overlap_coefficient index: 0.59012651
    - Median overlap_coefficient index: 0.62500000
- Avg of HPO / EHR: 9 
    - Avg jaccard index: 0.27572123
    - Median jaccard index: 0.29411765
    - Avg overlap_coefficient index: 0.60092468
    - Median overlap_coefficient index: 0.66666667
- Avg of HPO / EHR: 10 
    - Avg jaccard index: 0.29775511
    - Median jaccard index: 0.31818182
    - Avg overlap_coefficient index: 0.61199475
    - Median overlap_coefficient index: 0.60000000
- Avg of HPO / EHR: 11 
    - Avg jaccard index: 0.32044096
    - Median jaccard index: 0.35000000
    - Avg overlap_coefficient index: 0.63154168
    - Median overlap_coefficient index: 0.63636364
- Avg of HPO / EHR: 12 
    - Avg jaccard index: 0.34114855
    - Median jaccard index: 0.36363636
    - Avg overlap_coefficient index: 0.64963788
    - Median overlap_coefficient index: 0.66666667
- Avg of HPO / EHR: 13 
    - Avg jaccard index: 0.36223825
    - Median jaccard index: 0.39130435
    - Avg overlap_coefficient index: 0.66565314
    - Median overlap_coefficient index: 0.69230769
- Avg of HPO / EHR: 14 
    - Avg jaccard index: 0.38262520
    - Median jaccard index: 0.42105263
    - Avg overlap_coefficient index: 0.68512696
    - Median overlap_coefficient index: 0.71428571
- Avg of HPO / EHR: 15 
    - Avg jaccard index: 0.40238927
    - Median jaccard index: 0.43750000
    - Avg overlap_coefficient index: 0.70535461
    - Median overlap_coefficient index: 0.73333333
- Avg of HPO / EHR: 16 
    - Avg jaccard index: 0.42048832
    - Median jaccard index: 0.45833333
    - Avg overlap_coefficient index: 0.72891910
    - Median overlap_coefficient index: 0.75000000
- Avg of HPO / EHR: 17 
    - Avg jaccard index: 0.43893850
    - Median jaccard index: 0.50000000
    - Avg overlap_coefficient index: 0.74807863
    - Median overlap_coefficient index: 0.76470588
- Avg of HPO / EHR: 18 
    - Avg jaccard index: 0.45696400
    - Median jaccard index: 0.50000000
    - Avg overlap_coefficient index: 0.77689392
    - Median overlap_coefficient index: 0.77777778
- Avg of HPO / EHR: 19 
    - Avg jaccard index: 0.47406370
    - Median jaccard index: 0.52173913
    - Avg overlap_coefficient index: 0.80329467
    - Median overlap_coefficient index: 0.80000000
- Avg of HPO / EHR: 20 
    - Avg jaccard index: 0.49095388
    - Median jaccard index: 0.54166667
    - Avg overlap_coefficient index: 0.83769304
    - Median overlap_coefficient index: 0.84210526
- Avg of HPO / EHR: 21 
    - Avg jaccard index: 0.50698960
    - Median jaccard index: 0.56521739
    - Avg overlap_coefficient index: 0.87760788
    - Median overlap_coefficient index: 0.88235294
- Avg of HPO / EHR: 22 
    - Avg jaccard index: 0.52338340
    - Median jaccard index: 0.58333333
    - Avg overlap_coefficient index: 0.91822944
    - Median overlap_coefficient index: 0.90909091
- Avg of HPO / EHR: 23 
    - Avg jaccard index: 0.53923833
    - Median jaccard index: 0.58333333
    - Avg overlap_coefficient index: 0.95860996
    - Median overlap_coefficient index: 0.95238095
- Avg of HPO / EHR: 24 
    - Avg jaccard index: 0.55449478
    - Median jaccard index: 0.62500000
    - Avg overlap_coefficient index: 1.00000000
    - Median overlap_coefficient index: 1.00000000
                

###### Original ICD 2 HPO

Baselines: Jaccard Index (Testing set)
- Original ICD 2 HPO
- ICD2HPO vs. key: 0.27225081
- ICD2HPO vs. OBO: 0.28166703
- ICD2HPO vs. NCBO: 0.28436534
- key vs. OBO: 0.62668099
- key vs. NCBO: 0.68342156
- OBO vs. NCBO: 0.82151439

Baselines： Precision (Testing set)
- Original ICD 2 HPO
- silver: Intersection of (key, OBO, NCBO)
- silver vs. key: 0.79738606
- silver vs. NCBO: 0.71841184
- silver vs. OBO: 0.70158939 

Baselines： Recall (Testing set)
- Original ICD 2 HPO
- silver: Union of (key, OBO, NCBO)
- silver vs. key: 0.74574075
- silver vs. NCBO: 0.82367081
- silver vs. OBO: 0.84583867

Statistics:
- silver: Intersection of (key, OBO, NCBO)
    - Num of EHR has HPO 52438/52722
    - Avg HPO/EHR for all 7
    - Median HPO/EHR for all 7
    - Avg HPO/EHR for those have 7
    - Median HPO/EHR for those have 7
- silver: Union of (key, OBO, NCBO)
    - Avg HPO/EHR for all 12
    - Median HPO/EHR for all 13
    - Avg HPO/EHR for those have 12
    - Median HPO/EHR for those have 13


##### Evaluation on ICD

Evalution using ICD (FULL ICD CODE):
- Apply the reverse mapping HPO -> OMIM -> ICD (method #1) and get ICD results
- ALL MIMIC
- ICD2HPO 
    - Avg jaccard index: 0.02028501
    - Median jaccard index: 0.01724138
    - Avg overlap_coefficient index: 1.00000000
    - Median overlap_coefficient index: 1.00000000
    - Avg precision index: 0.02028501
    - Median precision index: 0.01724138
    - Avg recall index: 1.00000000
    - Median recall index: 1.00000000
- keyword
    - Avg jaccard index: 0.01691733
    - Median jaccard index: 0.00000000
    - Avg overlap_coefficient index: 0.35735850
    - Median overlap_coefficient index: 0.00000000
    - Avg precision index: 0.01713766
    - Median precision index: 0.00000000
    - Avg recall index: 0.35735850
    - Median recall index: 0.00000000
- ncbo
    - Avg jaccard index: 0.02170225
    - Median jaccard index: 0.02325581
    - Avg overlap_coefficient index: 0.54991568
    - Median overlap_coefficient index: 0.50000000
    - Avg precision index: 0.02190450
    - Median precision index: 0.02325581
    - Avg recall index: 0.54989774
    - Median recall index: 0.50000000
- obo
    - Avg jaccard index: 0.01987377
    - Median jaccard index: 0.02173913
    - Avg overlap_coefficient index: 0.51250477
    - Median overlap_coefficient index: 0.50000000
    - Avg precision index: 0.02003963
    - Median precision index: 0.02222222
    - Avg recall index: 0.51250477
    - Median recall index: 0.50000000
    
Evalution using ICD (ONLY FIRST 3-DIGIT of ICD CODE):
- Apply the reverse mapping HPO -> OMIM -> 3-DIGIT ICD (method #1) and get 3-DIGIT ICD results
- 65 3-digit ICD codes have HPO mapping and all of them appear in MIMIC
- MIMIC originally has 6918 3-digit ICD codes
- ALL MIMIC
- silver
    - Avg jaccard index: 0.04592643
    - Median jaccard index: 0.04761905
    - Avg overlap_coefficient index: 1.00000000
    - Median overlap_coefficient index: 1.00000000
    - Avg precision index: 0.04592643
    - Median precision index: 0.04761905
    - Avg recall index: 1.00000000
    - Median recall index: 1.00000000
- keyword
    - Avg jaccard index: 0.05163031
    - Median jaccard index: 0.04545455
    - Avg overlap_coefficient index: 0.34779038
    - Median overlap_coefficient index: 0.33333333
    - Avg precision index: 0.05833553
    - Median precision index: 0.05000000
    - Avg recall index: 0.34754203
    - Median recall index: 0.33333333
- ncbo
    - Avg jaccard index: 0.05173297
    - Median jaccard index: 0.04761905
    - Avg overlap_coefficient index: 0.44728759
    - Median overlap_coefficient index: 0.50000000
    - Avg precision index: 0.05561317
    - Median precision index: 0.05000000
    - Avg recall index: 0.44715160
    - Median recall index: 0.50000000
- obo
    - Avg jaccard index: 0.04843394
    - Median jaccard index: 0.04347826
    - Avg overlap_coefficient index: 0.41384477
    - Median overlap_coefficient index: 0.40000000
    - Avg precision index: 0.05212230
    - Median precision index: 0.04761905
    - Avg recall index: 0.41374439
    - Median recall index: 0.40000000
