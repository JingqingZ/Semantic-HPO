## The code in this file is used for early-stage analysis

import config
import dataloader
from tqdm import tqdm

def orphadata():
    data_folder = "../data/orphadata/"
    hpo_file = data_folder + "en_product4_HPO.xml"

    import xml.etree.ElementTree as ET
    tree = ET.parse(hpo_file)
    root = tree.getroot()

def icd_distribution_in_mimic():
    from collections import Counter

    mimic_data = dataloader.load_mimic()
    full_icd_list = mimic_data['ICD9_CODE_LIST'].tolist()

    icd2hpo = dataloader.get_icd_hpo_silver_mapping()

    def _icd_dist(icd_list):
        icd_counter = Counter()
        icd_has_hpo_counter = Counter()
        for icdstr in icd_list:
            icds = icdstr.split("/")
            for icd in icds:
                icd_counter[icd] += 1
                if icd in icd2hpo:
                    icd_has_hpo_counter[icd] += 1
        return icd_counter, icd_has_hpo_counter

    icd_train_counter, icd_has_hpo_train_counter = _icd_dist([full_icd_list[index] for index in config.mimic_train_indices])
    icd_test_counter, icd_has_hpo_test_counter = _icd_dist([full_icd_list[index] for index in config.mimic_test_indices])
    print(icd_train_counter.most_common(50))
    print(icd_test_counter.most_common(50))
    print(icd_has_hpo_train_counter.most_common(50))
    print(icd_has_hpo_test_counter.most_common(50))

def num_mimic_negation():
    mimic_data = dataloader.load_mimic()
    mimic_doc = mimic_data['CLEAN_TEXT'].tolist()
    negative_word = ['not', 'never', 'hardly', 'rarely', 'rule out',
                     'ruled out', 'no', 'no evidence', 'no support', 'no suspicion',
                     'deny', 'denies', 'denied', 'unlikely', 'no change', 'without',
                     'no signs of', 'absence', 'not demonstrated', 'negative', 'doubt',
                     'versus', 'negation', 'no further', 'without any further', 'without further']
    negative_word = [' ' + w + ' ' for w in negative_word]
    negative_num = [0 for _ in negative_word]
    for doc in tqdm(mimic_doc):
        for widx, nword in enumerate(negative_word):
            negative_num[widx] += 1 if nword in doc else 0
    print("Total doc %d" % len(mimic_doc))
    print("Negation %s" % ", ".join(["%s:%d" % (nword, negative_num[widx]) for widx, nword in enumerate(negative_word)]))
    '''
    Total doc 52722
    Negation  not :44867,  never :6730,  hardly :31,  rarely :803,  rule out :5183,  ruled out :5201,  no :52134,  
    no evidence :19646,  no support :6,  no suspicion :30,  deny :210,  denies :22140,  denied :7377,  unlikely :3105,  
    no change :2930,  without :34670,  no signs of :2859,  absence :3117,  not demonstrated :64,  negative :31487,  
    doubt :170,  versus :4238,  negation :0,  no further :6168,  without any further :212,  without further :660

    '''


if __name__ == '__main__':
    # icd_distribution_in_mimic()
    num_mimic_negation()
