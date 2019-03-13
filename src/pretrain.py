
import os
import logging
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForPreTraining

import config
import dataloader
from dataset import BERTDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    ## Settings
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)

    ## create dataset
    corpus_data = dataloader.get_corpus()
    tokenizer = BertTokenizer(dataloader.vocab_file, do_lower_case=True)

    train_dataset = BERTDataset(corpus_data, tokenizer, seq_len=config.sequence_length)

    num_train_steps = int(
        len(train_dataset) / config.train_batch_size * config.train_epoch
    )


    exit()

    model = BertForPreTraining.from_pretrained(config.bert_model)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

if __name__ == '__main__':
    main()
