# Unsupervised Annotation of Phenotypic Abnormalities via Semantic Latent Representations on Electronic Health Records. BIBM 2019. Regular Paper Accepted.

#### Jingqing Zhang, Xiaoyu Zhang, Kai Sun, Xian Yang, Chengliang Dai, and Yike Guo

Paper link: [arXiv:1911.03862](https://arxiv.org/abs/1911.03862)

## Abstract
The extraction of phenotype information which is naturally contained in 
electronic health records (EHRs) has been found to be useful in various clinical 
informatics applications such as disease diagnosis. However, due to imprecise 
descriptions, lack of gold standards and the demand for efficiency, annotating 
phenotypic abnormalities on millions of EHR narratives is still challenging. 
In this work, we propose a novel unsupervised deep learning framework to 
annotate the phenotypic abnormalities from EHRs via semantic latent representations. 
The proposed framework takes the advantage of Human Phenotype Ontology (HPO), 
which is a knowledge base of phenotypic abnormalities, to standardize the annotation 
results. Experiments have been conducted on 52,722 EHRs from MIMIC-III dataset. 
Quantitative and qualitative analysis have shown the proposed framework achieves 
state-of-the-art annotation performance and computational efficiency compared with 
other methods.

## Prerequisites
- Python 3.5+
- PyTorch 1.0+
- Others:
    - numpy, pandas, tqdm
    - etc.

## Code Structure
- [config.py](src/config.py): hyper-parametres, path configuration
- [dataloader.py](src/dataloader.py): path config of data, sources of raw data, preprocessing of data
    - Most intermediate data files should be created automatically as long as the raw data is provided. Please submit an issue if not.
- [dataset.py](src/dataset.py): processing of data
- [decision.py](src/decision.py): annotation strategy
- [evaluation.py](src/evaluation.py): evaluation metrics
- [loss_func.py](src/loss_func.py): loss functions
- [models.py](src/models.py): models
    - check Encoder, Generator, PriorConstraintModel
    - others are deprecated.
- [train.py](src/train.py): controller of train and test
    - check UnsupervisedAnnotationController
    - others are deprecated.


## Acknowledgement
Jingqing Zhang would like to thank the support from 
[LexisNexis Risk Solutions HPCC Systems academic program](https://hpccsystems.com/community/academics/imperial-college-london).
The authors would also like to thank the support from [Pangaea Data](https://www.pangaeaenterprises.co.uk/).
