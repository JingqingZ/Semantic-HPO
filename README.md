# Medical Text

### To Kai & Chengliang

* analyze ICD->HPO vs. Keyword Search
* run `python evaluation.py` to get worst / best cases, e.g., in printout
    - `0 (385, 0.0)`
    - `['HP:0025031', 'HP:0000119', 'HP:0001939', 'HP:0002086']`
    - `['HP:0001871', 'HP:0001626', 'HP:0025142', 'HP:0000707', 'HP:0002715']`
    - This is an example of worst case. `385` is the EHR id. `0.0` is the Jaccard.
    - The first line is the results of ICD->HPO. (unsorted)
    - The second line is the results of keyword searomim2hpo.jsonch. (unsorted)
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

Baselines: Jaccard Index (Testing set)
- ICD2HPO vs. key: 0.27225081
- ICD2HPO vs. OBO: 0.28166703
- ICD2HPO vs. NCBO: 0.28436534
- key vs. OBO: 0.62668099
- key vs. NCBO: 0.68342156
- OBO vs. NCBO: 0.82151439

Baselines： Precision (Testing set)
- silver: Intersection of (key, OBO, NCBO)
- silver vs. key: 0.79738606
- silver vs. NCBO: 0.71841184
- silver vs. OBO: 0.70158939 

Baselines： Recall (Testing set)
- silver: Union of (key, OBO, NCBO)
- silver vs. key: 0.74574075
- silver vs. NCBO: 0.82367081
- silver vs. OBO: 0.84583867
