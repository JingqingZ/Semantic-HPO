
import os
import logging
from tqdm import tqdm, trange
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertForPreTraining, BertAdam
from pytorch_pretrained_bert import BertModel, BertForNextSentencePrediction, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import warmup_linear

import config
import dataloader
from models import BertForSentenceEmbedding
from dataset import BERTDataset, HPOAnnotate4TrainingDataset, SentenceEmbeddingDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnotationSuperModel():

    def __init__(self):

        ## settings
        self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(self.device, self.n_gpu))

        ## tokenizer
        self.tokenizer = BertTokenizer(dataloader.vocab_file, do_lower_case=True)

        ## create bert model
        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=len(self.tokenizer.vocab),
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02
        )


    def pretrain(self, base_step=0, is_train=True):

        # create model for pretraining
        self.bert_model = BertForPreTraining(self.bert_config)
        self.bert_model.to(self.device)

        if self.n_gpu > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

        ## restore model if applicable
        if base_step > 0:
            logger.info("** ** * Restoring model from step %d ** ** * " % base_step)
            model_file = os.path.join(config.outputs_model_dir, "pytorch_model_epoch%d.bin" % (base_step))
            state_dict = torch.load(model_file)
            # state_dict = {key.replace("module.", ""): state_dict[key] for key in state_dict}
            self.bert_model.load_state_dict(state_dict, strict=True)

        ## weights
        weights = list(self.bert_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_weights = [
            {'params': [p for n, p in weights if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in weights if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if not is_train:
            assert base_step > 0

        ## create dataset
        hpo_ontology = dataloader.load_hpo_ontology()
        corpus_mimic_data, corpus_hpo_data = dataloader.get_corpus()
        assert len(corpus_mimic_data) == config.total_num_mimic_record

        if is_train:
            # train_corpus_mimic = corpus_mimic_data[:int(len(corpus_mimic_data) * config.training_percentage)]
            train_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_train_indices]
            corpus_dataset = BERTDataset(train_corpus_mimic, self.tokenizer,
                                         corpus_hpo_data, hpo_ontology, seq_len=config.sequence_length)
            corpus_sampler = RandomSampler(corpus_dataset)
            corpus_dataloader = DataLoader(corpus_dataset, sampler=corpus_sampler, batch_size=config.train_batch_size)
            total_num_steps = int(
                len(corpus_dataset) / config.train_batch_size * config.train_epoch
            )
            total_num_epoch = config.train_epoch

        else:
            del hpo_ontology
            del corpus_hpo_data
            # test_corpus_mimic = corpus_mimic_data[-int(len(corpus_mimic_data) * config.testing_percentage):]
            test_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_test_indices]
            corpus_dataset = BERTDataset(test_corpus_mimic, self.tokenizer, seq_len=config.sequence_length)
            corpus_dataloader = DataLoader(corpus_dataset, batch_size=config.test_batch_size)
            total_num_steps = int(
                len(corpus_dataset) / config.test_batch_size
            )
            total_num_epoch = 1

        if is_train:
            optimizer = BertAdam(
                optimizer_grouped_weights,
                lr=config.learning_rate_4_pretrain,
                warmup=config.warmup_proportion,
                t_total=total_num_steps
            )

        ## start training
        global_step = 0
        logger.info("***** Running %s *****" % ('training' if is_train else 'testing'))
        logger.info("  Num examples = %d", len(corpus_dataset))
        logger.info("  Batch size = %d", config.train_batch_size if is_train else config.test_batch_size)
        logger.info("  Num steps = %d", total_num_steps)

        if is_train:
            self.bert_model.train()
        else:
            self.bert_model.eval()

        for cur_epoch in trange(total_num_epoch, desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # for step, batch in enumerate(train_dataloader):
            for step, batch in enumerate(tqdm(corpus_dataloader, desc="Iteration")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                loss = self.bert_model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)

                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if global_step % config.print_step == 0:
                    logger.info("Step: %d, Total Step: %d, loss: %.4f" % (global_step, global_step + base_step, tr_loss / nb_tr_steps))

                if is_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                global_step += 1

                ## Save a trained model
                if is_train and global_step % config.save_step_4_pretrain == 0:
                    logger.info("** ** * Saving model at step %d ** ** * " % (base_step + global_step))
                    model_to_save = self.bert_model
                    output_model_file = os.path.join(config.outputs_model_dir, "pytorch_model_epoch%d.bin" %
                                                     (base_step + global_step))
                    torch.save(model_to_save.state_dict(), output_model_file)

    def embedding_space_analysis(self, pretrained_base_model, corpus_to_analysis):
        assert pretrained_base_model > 0

        self.bert_model = BertForSentenceEmbedding(self.bert_config)
        self.bert_model.to(self.device)

        if self.n_gpu > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

        logger.info("** ** * Restoring model from step %d ** ** * " % pretrained_base_model)
        model_file = os.path.join(config.outputs_model_dir, "pytorch_model_epoch%d.bin" % (pretrained_base_model))
        state_dict = torch.load(model_file)
        # state_dict = {key.replace("module.", ""): state_dict[key] for key in state_dict}
        self.bert_model.load_state_dict(state_dict, strict=False)

        whole_corpus_mimic_data, corpus_hpo_data = dataloader.get_corpus()
        if corpus_to_analysis == 'hpo':
            hpo_sentences = sorted(corpus_hpo_data.items(), key=lambda k: k[0])
            sentence_list = [t[1] for t in hpo_sentences]
        elif corpus_to_analysis == 'train':
            corpus_mimic = [whole_corpus_mimic_data[index] for index in config.mimic_train_indices]
            sentence_list = dataloader.get_sentence_list_mimic(corpus_mimic)
        elif corpus_to_analysis == 'test':
            corpus_mimic = [whole_corpus_mimic_data[index] for index in config.mimic_test_indices]
            sentence_list = dataloader.get_sentence_list_mimic(corpus_mimic)
        elif corpus_to_analysis == 'all':
            corpus_mimic = [whole_corpus_mimic_data[index] for index in np.append(config.mimic_train_indices, config.mimic_test_indices)]
            sentence_list = dataloader.get_sentence_list_mimic(corpus_mimic)
        else:
            raise ValueError('Invalid argument mimic_corpus_to_analysis.')

        corpus_dataset = SentenceEmbeddingDataset(sentence_list, self.tokenizer, seq_len=config.sequence_length)
        corpus_dataloader = DataLoader(corpus_dataset, batch_size=config.test_batch_size)

        self.bert_model.eval()

        embedding_list = list()
        for step, batch in enumerate(tqdm(corpus_dataloader, desc="Iteration")):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            pooled_output = self.bert_model(input_ids, segment_ids, input_mask)
            embedding_list.append(pooled_output.detach().cpu())

        embedding_list = torch.cat(embedding_list)
        embedding_list = embedding_list.numpy()

        print("Num of sentences %s" % len(sentence_list))
        print("Sentence embedding ", embedding_list.shape)
        with open(config.outputs_results_dir + "mimic_sentence_%s.pickle" % corpus_to_analysis, 'wb') as f:
            pickle.dump(sentence_list, f)
        with open(config.outputs_results_dir + "mimic_embedding_%s.npy" % corpus_to_analysis, 'wb') as f:
            np.save(f, embedding_list)

    def annotation_training(self, base_step, pretrained_base_model):

        assert pretrained_base_model > 0 or base_step > 0

        # create model
        self.bert_model = BertForSequenceClassification(self.bert_config, num_labels=2)
        self.bert_model.to(self.device)

        if self.n_gpu > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

        ## restore model if applicable
        if base_step > 0:
            logger.info("** ** * Restoring model from step %d ** ** * " % base_step)
            model_file = os.path.join(config.outputs_model_dir, "annotation_model_epoch%d.bin" % (base_step))
            self.bert_model.load_state_dict(torch.load(model_file), strict=True)
        elif pretrained_base_model > 0:
            logger.info("** ** * Restoring pretrained model from step %d ** ** * " % pretrained_base_model)
            model_file = os.path.join(config.outputs_model_dir, "pytorch_model_epoch%d.bin" % (pretrained_base_model))
            self.bert_model.load_state_dict(torch.load(model_file), strict=False)
        else:
            raise Exception("Invalid base_step or pretrained_base_model.")

        ## weights
        weights = list(self.bert_model.named_parameters())

        classifier_weights = [
               {'params': [p for n, p in weights if "module.classifier" in n], 'weight_decay': 0.0},
        ]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_weights = [
            {'params': [p for n, p in weights if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in weights if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        ## datasets and dataloader
        hpodata = dataloader.get_hpo4dataset()
        ann_dataset = HPOAnnotate4TrainingDataset(hpodata, tokenizer=self.tokenizer, seq_len=config.sequence_length)
        exit()

        ann_sampler = RandomSampler(ann_dataset)
        ann_dataloader = DataLoader(ann_dataset, sampler=ann_sampler, batch_size=config.train_batch_size)
        total_num_steps = int(
            len(ann_dataset) / config.train_batch_size * config.train_epoch
        )
        total_num_epoch = config.train_epoch

        ## optimizers
        optimizer_classifier = BertAdam(
            classifier_weights,
            lr=config.learning_rate_4_annotation,
            warmup=config.warmup_proportion,
            t_total=total_num_steps
        )

        optimizer_all = BertAdam(
            optimizer_grouped_weights,
            lr=config.learning_rate_4_pretrain,
            warmup=config.warmup_proportion,
            t_total=total_num_steps
        )


        ## start training
        global_step = 0
        logger.info("***** Running %s *****" % ('training' if True else 'testing'))
        logger.info("  Num examples = %d", len(ann_dataset))
        logger.info("  Batch size = %d", config.train_batch_size if True else config.test_batch_size)
        logger.info("  Num steps = %d", total_num_steps)

        self.bert_model.train()

        for cur_epoch in trange(total_num_epoch, desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # for step, batch in enumerate(train_dataloader):
            for step, batch in enumerate(tqdm(ann_dataloader, desc="Iteration")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, is_related = batch
                loss = self.bert_model(input_ids, segment_ids, input_mask, is_related)

                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if global_step % config.print_step == 0:
                    logger.info("Step: %d, Total Step: %d, loss: %.4f" % (global_step, global_step + base_step, tr_loss / nb_tr_steps))

                loss.backward()

                if global_step + base_step > config.step_to_train_all_weights:
                    optimizer_all.step()
                    optimizer_all.zero_grad()
                else:
                    optimizer_classifier.step()
                    optimizer_classifier.zero_grad()

                global_step += 1

                ## Save a trained model
                if global_step % config.save_step_4_annotation == 0:
                    logger.info("** ** * Saving model at step %d ** ** * " % (base_step + global_step))
                    model_to_save = self.bert_model
                    output_model_file = os.path.join(config.outputs_model_dir, "annotation_model_epoch%d.bin" %
                                                     (base_step + global_step))
                    torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == '__main__':

    supermodel = AnnotationSuperModel()

    # supermodel.pretrain(base_step=0, is_train=True)

    # supermodel.pretrain(base_step=100000, is_train=False) # loss=1.9213, 1.819 (100 steps)
    # supermodel.pretrain(base_step=150000, is_train=False) # loss=2.1155, 1.655 (100 steps)
    # supermodel.pretrain(base_step=200000, is_train=False) # loss=1.7342, 1.534 (100 steps)

    supermodel.embedding_space_analysis(pretrained_base_model=200000, corpus_to_analysis='hpo')
    # supermodel.embedding_space_analysis(pretrained_base_model=200000, corpus_to_analysis='test')
    supermodel.embedding_space_analysis(pretrained_base_model=200000, corpus_to_analysis='train')

    # supermodel.annotation_training(base_step=0, pretrained_base_model=360000)

    pass
