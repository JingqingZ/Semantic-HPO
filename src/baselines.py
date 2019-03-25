
import dataloader

def silver_standard():
    # mimic_data = dataloader.load_mimic()
    icd2omim = dataloader.get_icd_omim_mapping()
    omim2hpo = dataloader.get_omim_hpo_mapping()

    for icd in icd2omim:
        print(icd, icd2omim[icd])
        exit()

def keyword_search():
    pass

if __name__ == '__main__':
    silver_standard()
