
import os
import random
import logging
from tqdm import tqdm, trange
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertForPreTraining, BertAdam
from pytorch_pretrained_bert import BertModel, BertForNextSentencePrediction, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import warmup_linear

import utils
import config
import dataloader
import models
import loss_func
from models import BertForSentenceEmbedding
from dataset import BERTDataset, HPOAnnotate4TrainingDataset, SentenceEmbeddingDataset, \
    UnsupervisedAnnotationMIMICDataset, UnsupervisedAnnotationHPODataset, UnsupervisedAnnotationHPORootDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# deprecated
class AnnotationSuperModel():

    def __init__(self):

        ## settings
        self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(self.device, self.n_gpu))

        ## tokenizer
        self.tokenizer = BertTokenizer(dataloader.vocab_file, do_lower_case=True)
        print(len(self.tokenizer.vocab))
        exit()

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
            _, pooled_output = self.bert_model(input_ids, segment_ids, input_mask)
            embedding_list.append(pooled_output.detach().cpu())

            # print(input_ids.shape)
            # print(pooled_output.shape)
            # print(_.shape)
            # exit()

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

class UnsupervisedAnnotationController():

    def __init__(self):

        ## settings
        self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(self.device, self.n_gpu))

        ## tokenizer
        self.tokenizer = BertTokenizer(dataloader.vocab_file, do_lower_case=True)

        ## create bert model
        self.config = models.Config(
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.01,
            max_seq_len=config.sequence_length // 2,
            num_hpo_node=len(dataloader.hpo_limited_list),
            hpo_hidden_size=1536,
            vocab_size=len(self.tokenizer.vocab)
        )
        # TODO: CNN
        self.cnnconfig = models.CNNConfig(
            embed_hidden_size=768,
            max_position_embeddings=512,
            max_seq_len=32,
            num_hpo_node=len(dataloader.hpo_limited_list),
            hpo_hidden_size=1536,
            vocab_size=len(self.tokenizer.vocab)
        )

    def train(self, base_step=0, is_train=True):

        # create model for pretraining

        self.encoder = models.Encoder(self.config)
        self.generator = models.Generator(self.config)
        # TODO: CNN
        # self.encoder = models.EncoderCNN(self.cnnconfig)
        # self.generator = models.GeneratorCNN(self.cnnconfig)

        self.prior_constraint_model = models.PriorConstraintModel(self.config)

        self.encoder.to(self.device)
        self.generator.to(self.device)
        self.prior_constraint_model.to(self.device)

        if self.n_gpu > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.generator = torch.nn.DataParallel(self.generator)
            self.prior_constraint_model = torch.nn.DataParallel(self.prior_constraint_model)

        ## restore model if applicable
        if base_step > 0:
            logger.info("** ** * Restoring model from step %d ** ** * " % base_step)
            success = False
            encoder_model_file = os.path.join(config.outputs_model_dir, "encoder_epoch%d.bin" % (base_step))
            generator_model_file = os.path.join(config.outputs_model_dir, "generator_epoch%d.bin" % (base_step))
            prior_model_file = os.path.join(config.outputs_model_dir, "prior_constraint_model_epoch%d.bin" % (base_step))
            try:
                utils.load_model(self.encoder, encoder_model_file)
                utils.load_model(self.generator, generator_model_file)
                utils.load_model(self.prior_constraint_model, prior_model_file)
                success = True
            except:
                pass
            try:
                if not success:
                    utils.load_model_rm_module(self.encoder, encoder_model_file)
                    utils.load_model_rm_module(self.generator, generator_model_file)
                    utils.load_model_rm_module(self.prior_constraint_model, prior_model_file)
                success = True
            except:
                pass
            try:
                if not success:
                    utils.load_model_add_module(self.encoder, encoder_model_file)
                    utils.load_model_add_module(self.generator, generator_model_file)
                    utils.load_model_add_module(self.prior_constraint_model, prior_model_file)
                success = True
            except:
                pass
            if not success:
                logger.error("FAIL TO LOAD MODEL ...")
            else:
                logger.info("MODELS RESTORED ...")

        ## weights
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        encoder_weights = list(self.encoder.named_parameters())
        generator_weights = list(self.generator.named_parameters())
        prior_constraint_model_weights = list(self.prior_constraint_model.named_parameters())
        
        # res_weights = encoder_weights + generator_weights
        # pr_weights = encoder_weights + prior_constraint_model_weights
        all_weights = encoder_weights + generator_weights + prior_constraint_model_weights

        # optimizer_grouped_res_weights = [
        #     {'params': [p for n, p in res_weights if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in res_weights if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer_grouped_pr_weights = [
        #     {'params': [p for n, p in pr_weights if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in pr_weights if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        optimizer_grouped_weights = [
            {'params': [p for n, p in all_weights if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in all_weights if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if not is_train:
            assert base_step > 0

        ## create dataset
        # hpo_ontology = dataloader.load_hpo_ontology()
        corpus_mimic_data, corpus_hpo_data = dataloader.get_corpus()
        hpo_children_info = dataloader.get_hpo_children_info()
        # hpo_description_list = [corpus_hpo_data[hpo] for hpo in dataloader.hpo_limited_list]
        assert len(corpus_mimic_data) == config.total_num_mimic_record

        hpo_corpus_dataset = UnsupervisedAnnotationHPODataset(
            dataloader.hpo_limited_list,
            hpo_children_info,
            corpus_hpo_data,
            self.tokenizer, seq_len=config.sequence_length // 2
        )
        hpo_corpus_dataloader = DataLoader(
            hpo_corpus_dataset,
            sampler=RandomSampler(hpo_corpus_dataset) if is_train else None,
            batch_size=config.train_batch_size if is_train else config.test_batch_size
        )
        hpo_root_dataset = UnsupervisedAnnotationHPORootDataset(
            dataloader.hpo_limited_list,
            corpus_hpo_data,
            self.tokenizer, seq_len=config.sequence_length // 2
        )
        # TODO: del to release memeory
        # del hpo_children_info, corpus_hpo_data

        if is_train:
            # train_corpus_mimic = corpus_mimic_data[:int(len(corpus_mimic_data) * config.training_percentage)]
            train_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_train_indices]
            train_mimic_sentence_list = [
                sentence
                for doc in train_corpus_mimic for sentence in doc.split("\n") if len(sentence) > 0
            ]
            mimic_corpus_dataset = UnsupervisedAnnotationMIMICDataset(
                # train_corpus_mimic,
                train_mimic_sentence_list,
                self.tokenizer, seq_len=config.sequence_length // 2
            )
            mimic_corpus_sampler = RandomSampler(mimic_corpus_dataset)
            mimic_corpus_dataloader = DataLoader(mimic_corpus_dataset, sampler=mimic_corpus_sampler, batch_size=config.train_batch_size)
            total_num_steps = int(
                len(mimic_corpus_dataset) / config.train_batch_size * config.train_epoch_unsupervised
            )
            total_num_epoch = config.train_epoch_unsupervised

        else:
            # test_corpus_mimic = corpus_mimic_data[-int(len(corpus_mimic_data) * config.testing_percentage):]
            test_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_test_indices]
            test_mimic_sentence_list = [
                sentence
                for doc in test_corpus_mimic for sentence in doc.split("\n") if len(sentence) > 0
            ]
            mimic_corpus_dataset = UnsupervisedAnnotationMIMICDataset(
                # test_corpus_mimic,
                test_mimic_sentence_list,
                self.tokenizer, seq_len=config.sequence_length // 2
            )
            mimic_corpus_dataloader = DataLoader(mimic_corpus_dataset, batch_size=config.test_batch_size)
            total_num_steps = int(
                len(mimic_corpus_dataset) / config.test_batch_size
            )
            total_num_epoch = 1

        if is_train:
            # optimizer = BertAdam(
            #     optimizer_grouped_res_weights,
            #     lr=config.learning_rate_4_unsupervised,
            #     warmup=0,
            #     t_total=total_num_steps
            # )
            # optimizer = optim.Adam(
            #     optimizer_grouped_res_weights,
            #     lr=config.learning_rate_4_unsupervised,
            # )
            # optimizer_prime = optim.Adam(
            #     optimizer_grouped_pr_weights,
            #     lr=config.learning_rate_4_unsupervised,
            # )
            optimizer = optim.Adam(
                optimizer_grouped_weights,
                lr=config.learning_rate_4_unsupervised,
            )
        loss_function = loss_func.Loss()

        ## start training
        global_step = 0
        logger.info("***** Running %s *****" % ('training' if is_train else 'testing'))
        logger.info("  Num examples = %d", len(mimic_corpus_dataset))
        logger.info("  Batch size = %d", config.train_batch_size if is_train else config.test_batch_size)
        logger.info("  MIMIC num step / epoch = %d", len(mimic_corpus_dataloader))
        logger.info("  HPO num step / epoch = %d", len(hpo_corpus_dataloader))

        if is_train:
            self.encoder.train()
            self.generator.train()
            self.prior_constraint_model.train()
        else:
            self.encoder.eval()
            self.generator.eval()
            self.prior_constraint_model.eval()

        hpo_ids = list()
        for hsample in hpo_root_dataset:
            hpo_ids.append(hsample[0])
        hpo_ids = torch.stack(hpo_ids, dim=0)
        assert hpo_ids.shape[0] == len(dataloader.hpo_limited_list)
        assert hpo_ids.shape[1] == config.sequence_length // 2

        for cur_epoch in range(total_num_epoch):

            def _hpo_res(cof1, cof2, cof3, cof4, percentage=1.0, single_step=False):

                res_hpo_loss_agg = res_alpha_loss_agg = prior_loss_agg = vector_classify_loss_agg = 0
                for step, batch in enumerate(hpo_corpus_dataloader):

                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_ids_hpos, input_mask, segment_ids, one_hpo_id, all_hpo_alpha = batch

                    #########
                    # prior constrain

                    alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)
                    # print(alpha_out.shape)
                    # print(all_hpo_latent_outputs.shape)

                    ## extract one hpo
                    one_hpo_alpha_out = np.zeros(
                        shape=alpha_out.shape
                    ).astype(np.float32)

                    # one_hpo_latent_outputs = np.zeros(
                    #     shape=[all_hpo_latent_outputs.size()[0], all_hpo_latent_outputs.size()[2]]
                    # ).astype(np.float32)

                    one_hpo_latent_outputs = list()
                    for i in range(one_hpo_id.shape[0]):
                        one_hpo_alpha_out[i][one_hpo_id[i]] = float(1)
                        # one_hpo_latent_outputs[i] = all_hpo_latent_outputs[i][one_hpo_id[i]].detach().cpu().numpy()
                        one_hpo_latent_outputs.append(all_hpo_latent_outputs[i][one_hpo_id[i]])

                    one_hpo_alpha_out = torch.tensor(one_hpo_alpha_out)
                    one_hpo_alpha_out = one_hpo_alpha_out.to(self.device)

                    # one_hpo_latent_outputs = torch.tensor(one_hpo_latent_outputs)
                    one_hpo_latent_outputs = torch.stack(one_hpo_latent_outputs, dim=0)
                    one_hpo_latent_outputs = one_hpo_latent_outputs.to(self.device)
                    ##

                    # TODO add noise: one_hpo_latent_outputs
                    noise = torch.randint_like(one_hpo_latent_outputs, low=900, high=1100) / 1000
                    latent_vector_classify_out = self.prior_constraint_model(one_hpo_latent_outputs * noise, None)
                    vector_classify_loss = loss_function.multi_label_cross_entropy(latent_vector_classify_out, one_hpo_id)

                    noise = torch.randint_like(all_hpo_latent_outputs, low=950, high=1050) / 1000
                    hpo_reconstructed_sequence = self.generator(one_hpo_alpha_out, all_hpo_latent_outputs * noise, None)
                    hpo_constrain_loss = loss_function.resconstruction(hpo_reconstructed_sequence, input_ids_hpos)

                    # print(latent_vector_classify_out.shape)
                    # print(hpo_reconstructed_sequence.shape)
                    # exit()

                    # print(one_hpo_id)
                    # print(one_hpo_alpha_out)
                    # print(one_hpo_alpha_out.shape)
                    # print(torch.argmax(one_hpo_alpha_out, dim=1))
                    # exit()

                    prior_loss = hpo_constrain_loss

                    loss1 = cof3 * prior_loss + cof4 * vector_classify_loss

                    if self.n_gpu > 1:
                        prior_loss = prior_loss.mean()
                        vector_classify_loss = vector_classify_loss.mean()
                        loss1 = loss1.mean()

                    prior_loss_agg += prior_loss.item()
                    vector_classify_loss_agg += vector_classify_loss.item()

                    if is_train and not single_step:
                        # loss1.backward(retain_graph=True)
                        loss1.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    #########
                    # reconstruction: encoding + generating
                    # TODO: keep or not
                    if not single_step:
                        alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)

                    full_reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, None)
                    # full_reconstructed_sequence = self.generator(all_hpo_alpha, all_hpo_latent_outputs, None)


                    # TODO: printout, to remove
                    # torch.set_printoptions(threshold=5000)
                    # tmp = F.sigmoid(alpha_out)
                    # print(tmp[:10])
                    # print(all_hpo_alpha[:10])
                    # print(torch.topk(tmp, 2, dim=1)[1][:10])
                    # print(torch.topk(all_hpo_alpha, 2, dim=1)[1][:10])
                    # exit()

                    res_hpo_loss = loss_function.resconstruction(full_reconstructed_sequence, input_ids)
                    res_alpha_loss = loss_function.alpha_cross_entropy(alpha_out, all_hpo_alpha)

                    loss2 = cof1 * res_hpo_loss + cof2 * res_alpha_loss

                    if self.n_gpu > 1:
                        # res_loss = res_loss.mean() # mean() to average on multi-gpu.
                        res_hpo_loss = res_hpo_loss.mean()
                        res_alpha_loss = res_alpha_loss.mean()
                        loss2 = loss2.mean()

                    res_hpo_loss_agg += res_hpo_loss.item()
                    res_alpha_loss_agg += res_alpha_loss.item()

                    # training the reconstruction part
                    if is_train and not single_step:
                        loss2.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    if single_step:
                        if global_step % 100 == 0:
                            logger.info("[HPO] E: %d, Step: %d, loss: %.4f / %.4f / %.4f / %.4f" % (
                                cur_epoch, step, res_hpo_loss_agg / (step + 1), res_alpha_loss_agg / (step + 1),
                                prior_loss_agg / (step + 1), vector_classify_loss_agg / (step + 1)
                            ))
                        return loss1 + loss2

                    if (step > 0 or percentage > 0.99) and step % 100 == 0:
                    # if step % 100 == 0:
                        logger.info("[HPO] E: %d, Step: %d, loss: %.4f / %.4f / %.4f / %.4f" % (
                            cur_epoch, step, res_hpo_loss_agg / (step + 1), res_alpha_loss_agg / (step + 1),
                            prior_loss_agg / (step + 1), vector_classify_loss_agg / (step + 1)
                        ))

                    if step > len(hpo_corpus_dataloader) * percentage:
                        break

                if global_step % 100 == 0:
                    logger.info("[HPO] E: %d, Step: %d, loss: %.4f / %.4f / %.4f / %.4f" % (
                        cur_epoch, step, res_hpo_loss_agg / (step + 1), res_alpha_loss_agg / (step + 1),
                        prior_loss_agg / (step + 1), vector_classify_loss_agg / (step + 1)
                    ))

            # training on HPO descriptions firstly
            if global_step + base_step == 0:
                for _ in range(1):
                    _hpo_res(0.01, 10, 0.01, 2, percentage=1.0)

            # TODO: printout, to remove
            # _hpo_res(0.01, 10, 0.01, 2, percentage=1.0)
            # exit()

            tr_loss1 = tr_loss2 = 0
            tr_res_loss = 0
            tr_phe_loss = 0
            tr_vcl_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(mimic_corpus_dataloader, desc="MIMIC Iter")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids = batch
                # alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, input_mask)
                # full_reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, input_mask)
                alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)

                '''
                # TODO: printout, to remove
                torch.set_printoptions(threshold=5000)
                tmp = F.sigmoid(alpha_out)
                print(tmp[:10])
                print(torch.argmax(tmp, dim=1))
                counter = [0] * len(dataloader.hpo_limited_list)
                for i in range(tmp.shape[0]):
                    list_indices = list()
                    for j in range(tmp.shape[1]):
                        if tmp[i, j] > 0.1:
                            counter[j] += 1
                print(alpha_out.shape)
                hpo_children_info = dataloader.get_hpo_children_info()
                for i in range(len(counter)):
                    print(i, dataloader.hpo_limited_list[i], counter[i],
                          len(hpo_children_info[dataloader.hpo_limited_list[i]]))
                    print(corpus_hpo_data[dataloader.hpo_limited_list[i]])
                exit()
                '''

                ##########
                # prior constraint

                ## randomly select one hpo
                # which_hpo = np.random.randint(low=0, high=len(dataloader.hpo_limited_list), size=(alpha_out.shape[0]))
                which_hpo = np.array([random.randrange(0, len(dataloader.hpo_limited_list)) for _ in range(alpha_out.shape[0])])

                one_hpo_alpha_out = np.zeros(
                    shape=alpha_out.shape
                ).astype(np.float32)

                # one_hpo_latent_outputs = np.zeros(
                #     shape=[all_hpo_latent_outputs.size()[0], all_hpo_latent_outputs.size()[2]]
                # ).astype(np.float32)
                one_hpo_latent_outputs = list()

                hpo_input_ids = list()

                for i in range(which_hpo.shape[0]):
                    one_hpo_alpha_out[i][which_hpo[i]] = float(1)
                    # one_hpo_latent_outputs[i] = all_hpo_latent_outputs[i][which_hpo[i]].detach().cpu().numpy()
                    one_hpo_latent_outputs.append(all_hpo_latent_outputs[i][which_hpo[i]])
                    hpo_input_ids.append(hpo_ids[which_hpo[i]])

                which_hpo = torch.tensor(which_hpo)
                which_hpo = which_hpo.to(self.device)
                one_hpo_alpha_out = torch.tensor(one_hpo_alpha_out)
                one_hpo_alpha_out = one_hpo_alpha_out.to(self.device)
                hpo_input_ids = torch.stack(hpo_input_ids, dim=0)
                hpo_input_ids = hpo_input_ids.to(self.device)
                # one_hpo_latent_outputs = torch.tensor(one_hpo_latent_outputs)
                one_hpo_latent_outputs = torch.stack(one_hpo_latent_outputs, dim=0)
                one_hpo_latent_outputs = one_hpo_latent_outputs.to(self.device)
                ##

                # print(which_hpo)
                # print(which_hpo.shape)
                # print(one_hpo_alpha_out)
                # print(torch.argmax(one_hpo_alpha_out, dim=1))
                # print(one_hpo_alpha_out.shape)
                # print(one_hpo_latent_outputs.shape)
                # print(hpo_input_ids.shape)

                # add noise
                noise = torch.randint_like(one_hpo_latent_outputs, low=990, high=1010) / 1000
                latent_vector_classify_out = self.prior_constraint_model(one_hpo_latent_outputs * noise, None)
                vector_classify_loss = loss_function.multi_label_cross_entropy(latent_vector_classify_out, which_hpo)

                noise = torch.randint_like(all_hpo_latent_outputs, low=990, high=1010) / 1000
                hpo_reconstructed_sequence = self.generator(one_hpo_alpha_out, all_hpo_latent_outputs * noise, None)
                pr_loss = loss_function.resconstruction(hpo_reconstructed_sequence, hpo_input_ids)

                # loss1 = pr_loss + vector_classify_loss
                loss1 = vector_classify_loss

                if self.n_gpu > 1:
                    loss1 = loss1.mean() # mean() to average on multi-gpu.
                    pr_loss = pr_loss.mean()
                    vector_classify_loss = vector_classify_loss.mean()

                tr_loss1 += loss1.item()
                tr_phe_loss += pr_loss.item()
                tr_vcl_loss += vector_classify_loss.item()

                if is_train and global_step + base_step < 0:
                    # loss1.backward(retain_graph=True)
                    loss1.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                ##########
                # reconstruction: encoding + generating
                # TODO: keep or not
                if global_step + base_step < 0:
                    alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)

                full_reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, None)

                res_loss = loss_function.resconstruction(full_reconstructed_sequence, input_ids)

                loss2 = res_loss

                if self.n_gpu > 1:
                    loss2 = loss2.mean() # mean() to average on multi-gpu.
                    res_loss = res_loss.mean()

                tr_loss2 += loss2.item()
                tr_res_loss += res_loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if is_train and global_step + base_step < 0:
                    loss2.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                if global_step % config.print_step_unsupervised == 0:
                    logger.info("[MIMIC] E: %d, Step: %d, Total Step: %d, loss: %.4f / %.4f / %.4f" % (
                        cur_epoch, global_step, global_step + base_step,
                        tr_res_loss / nb_tr_steps, tr_phe_loss / nb_tr_steps, tr_vcl_loss / nb_tr_steps))

                if is_train and global_step + base_step >= 0:
                    # loss_hpo = _hpo_res(1, 10, 0.0, 0.1, single_step=True)
                    loss_hpo = _hpo_res(10, 2, 0.0, 0.1, single_step=True)
                    loss_new = loss1 + 10 * loss2 + loss_hpo
                    loss_new.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                else:
                    if global_step > 0 and global_step % 10 == 0:
                        # _hpo_res(0.5, 5, 0.5, 0.1,
                        _hpo_res(1, 5, 0.0, 0.1,
                                 percentage=0.01)

                global_step += 1

                ## Save a trained model
                if is_train and (
                    global_step + base_step == 1 or
                            (global_step + base_step) % config.save_step_4_unsupervised == 0
                ):
                    logger.info("** ** * Saving model at step %d ** ** * " % (base_step + global_step))
                    model_to_save = self.encoder
                    output_model_file = os.path.join(config.outputs_model_dir, "encoder_epoch%d.bin" %
                                                     (base_step + global_step))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save = self.generator
                    output_model_file = os.path.join(config.outputs_model_dir, "generator_epoch%d.bin" %
                                                     (base_step + global_step))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save = self.prior_constraint_model
                    output_model_file = os.path.join(config.outputs_model_dir, "prior_constraint_model_epoch%d.bin" %
                                                     (base_step + global_step))
                    torch.save(model_to_save.state_dict(), output_model_file)

    def inference(self, base_step, corpus_to_analysis='test'):

        # create model for pretraining
        self.encoder = models.Encoder(self.config)
        # self.encoder = models.EncoderCNN(self.config)
        self.encoder.to(self.device)
        # self.generator = models.Generator(self.config)
        # self.generator.to(self.device)

        if self.n_gpu > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
            # self.generator = torch.nn.DataParallel(self.generator)

        ## restore model if applicable
        ## if loading error: check https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
        assert base_step > 0
        logger.info("** ** * Restoring model from step %d ** ** * " % base_step)
        success = False
        encoder_model_file = os.path.join(config.outputs_model_dir, "encoder_epoch%d.bin" % (base_step))
        try:
            utils.load_model(self.encoder, encoder_model_file)
            success = True
        except:
            pass
        try:
            if not success:
                utils.load_model_rm_module(self.encoder, encoder_model_file)
            success = True
        except:
            pass
        try:
            if not success:
                utils.load_model_add_module(self.encoder, encoder_model_file)
            success = True
        except:
            pass
        if not success:
            logger.error("FAIL TO LOAD MODEL ...")
        else:
            logger.info("MODELS RESTORED ...")

        ## create dataset
        # hpo_ontology = dataloader.load_hpo_ontology()
        corpus_mimic_data, corpus_hpo_data = dataloader.get_corpus()
        hpo_description_list = [corpus_hpo_data[hpo] for hpo in dataloader.hpo_limited_list]
        assert len(corpus_mimic_data) == config.total_num_mimic_record

        hpo_root_dataset = UnsupervisedAnnotationHPORootDataset(
            dataloader.hpo_limited_list,
            corpus_hpo_data,
            self.tokenizer, seq_len=config.sequence_length // 2
        )

        hpo_ids = list()
        for hsample in hpo_root_dataset:
            hpo_ids.append(hsample[0])
        hpo_ids = torch.stack(hpo_ids, dim=0)
        assert hpo_ids.shape[0] == len(dataloader.hpo_limited_list)
        assert hpo_ids.shape[1] == config.sequence_length // 2

        if corpus_to_analysis == 'train':
            # train_corpus_mimic = corpus_mimic_data[:int(len(corpus_mimic_data) * config.training_percentage)]
            train_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_train_indices]
            train_mimic_sentence_list = [
                sentence
                for doc in train_corpus_mimic for sentence in doc.split("\n") if len(sentence) > 0
            ]
            mimic_corpus_dataset = UnsupervisedAnnotationMIMICDataset(
                train_mimic_sentence_list,
                self.tokenizer, seq_len=config.sequence_length // 2
            )
            mimic_corpus_dataloader = DataLoader(mimic_corpus_dataset, batch_size=config.test_batch_size)

        elif corpus_to_analysis == 'test':
            # test_corpus_mimic = corpus_mimic_data
            # test_corpus_mimic = corpus_mimic_data[-int(len(corpus_mimic_data) * config.testing_percentage):]
            test_corpus_mimic = [corpus_mimic_data[index] for index in config.mimic_test_indices]
            test_mimic_sentence_list = [
                sentence
                for doc in test_corpus_mimic for sentence in doc.split("\n") if len(sentence) > 0
            ]
            mimic_corpus_dataset = UnsupervisedAnnotationMIMICDataset(
                test_mimic_sentence_list,
                self.tokenizer, seq_len=config.sequence_length // 2
            )
            mimic_corpus_dataloader = DataLoader(mimic_corpus_dataset, batch_size=config.test_batch_size)
        else:
            raise Exception('Invalid corpus_to_analysis')

        total_num_steps = int(
            len(mimic_corpus_dataset) / config.test_batch_size
        )

        # loss_function = loss_func.Loss()

        ## start training
        logger.info("***** Running %s *****" % ('inference'))
        logger.info("  Num examples = %d", len(mimic_corpus_dataset))
        logger.info("  Batch size = %d", config.test_batch_size)
        logger.info("  Num steps = %d", total_num_steps)

        self.encoder.eval()

        '''
        #############
        # running on HPO descriptions firstly
        hpo_children_info = dataloader.get_hpo_children_info()
        hpo_corpus_dataset = UnsupervisedAnnotationHPODataset(
            dataloader.hpo_limited_list,
            hpo_children_info,
            corpus_hpo_data,
            self.tokenizer, seq_len=config.sequence_length // 2,
            duplication=False
        )
        hpo_corpus_dataloader = DataLoader(
            hpo_corpus_dataset,
            sampler=RandomSampler(hpo_corpus_dataset),
            batch_size=config.test_batch_size
        )

        hpo_alpha_results = []
        hpo_alpha_gt = []
        for step, batch in enumerate(tqdm(hpo_corpus_dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_ids_hpos, input_mask, segment_ids, one_hpo_id, all_hpo_alpha = batch

            alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)

            hpo_alpha_results.append(alpha_out.detach().cpu())
            hpo_alpha_gt.append(all_hpo_alpha.detach().cpu())

            # torch.set_printoptions(threshold=5000000)
            # print(torch.argmax(F.sigmoid(alpha_out), dim=1))
            # print(torch.argmax(all_hpo_alpha, dim=1))
            # exit()

        hpo_alpha_results = torch.cat(hpo_alpha_results, dim=0)
        hpo_alpha_results = hpo_alpha_results.numpy()
        hpo_alpha_gt = torch.cat(hpo_alpha_gt, dim=0)
        hpo_alpha_gt = hpo_alpha_gt.numpy()

        hpo_idx_list = [
            hpo_corpus_dataset.samples_with_duplication[i]["self_hpo_idx"]
            for i in range(len(hpo_corpus_dataset))
        ]

        print("HPO Num of samples: ", len(hpo_corpus_dataset))
        print("HPO Alpha size: ", hpo_alpha_results.shape)
        with open(config.outputs_results_dir + "hpo_idx_list.pickle", 'wb') as f:
            pickle.dump(hpo_idx_list, f)
        with open(config.outputs_results_dir + "hpo_alpha.npy", 'wb') as f:
            np.save(f, hpo_alpha_results)
        with open(config.outputs_results_dir + "hpo_alpha_gt.npy", 'wb') as f:
            np.save(f, hpo_alpha_gt)
        print("HPO Alpha saved")

        exit()
        ###############
        '''

        # tr_loss = 0
        # tr_res_loss = 0
        # tr_phe_loss = 0
        # nb_tr_examples, nb_tr_steps = 0, 0

        mimic_alpha_results = []
        for step, batch in enumerate(tqdm(mimic_corpus_dataloader, desc="MIMIC Iter")):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            # alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, input_mask)
            # reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, input_mask)
            alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)
            # reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, None)

            '''
            # TODO: comment this
            torch.set_printoptions(threshold=5000)
            tmp = F.sigmoid(alpha_out)
            print(tmp[:10])
            print(torch.argmax(tmp, dim=1))
            counter = [0] * len(dataloader.hpo_limited_list)
            for i in range(tmp.shape[0]):
                list_indices = list()
                for j in range(tmp.shape[1]):
                    if tmp[i, j] > 0.1:
                        counter[j] += 1
            print(alpha_out.shape)
            hpo_children_info = dataloader.get_hpo_children_info()
            for i in range(len(counter)):
                print(i, dataloader.hpo_limited_list[i], counter[i],
                      len(hpo_children_info[dataloader.hpo_limited_list[i]]))
            exit()
            '''

            mimic_alpha_results.append(alpha_out.detach().cpu())

            '''
            # randomly select hpos
            which_hpo = np.random.randint(low=0, high=len(dataloader.hpo_limited_list), size=(alpha_out.shape[0]))
            alpha_hpo = np.zeros(shape=alpha_out.size()).astype(np.float32)
            hpo_input_ids = list()
            for _ in range(alpha_hpo.shape[0]):
                alpha_hpo[_, which_hpo[_]] = 1
                hpo_input_ids.append(hpo_ids[which_hpo[_]])
            alpha_hpo = torch.tensor(alpha_hpo)
            alpha_hpo = alpha_hpo.to(self.device)
            hpo_input_ids = torch.stack(hpo_input_ids, dim=0)
            hpo_input_ids = hpo_input_ids.to(self.device)

            # reconstructed_hpo_desc = self.generator(alpha_hpo, all_hpo_latent_outputs, input_mask)
            reconstructed_hpo_desc = self.generator(alpha_hpo, all_hpo_latent_outputs, None)

            # print(all_hpo_latent_outputs.shape)
            # print(reconstructed_sequence.shape)

            res_loss = loss_function.resconstruction(reconstructed_sequence, input_ids)
            phe_loss = loss_function.resconstruction(reconstructed_hpo_desc, hpo_input_ids)
            loss = res_loss + phe_loss

            if self.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
                res_loss = res_loss.mean()
                phe_loss = phe_loss.mean()

            tr_loss += loss.item()
            tr_res_loss += res_loss.item()
            tr_phe_loss += phe_loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if step % config.print_step_unsupervised == 0:
                logger.info("Step: %d, BaseStep: %d, loss: %.4f / %.4f" % (
                    step, base_step,
                    tr_res_loss / nb_tr_steps, tr_phe_loss / nb_tr_steps))
            '''

        mimic_alpha_results = torch.cat(mimic_alpha_results, dim=0)
        mimic_alpha_results = mimic_alpha_results.numpy()

        print("MIMIC Num of samples: ", len(mimic_corpus_dataset))
        print("MIMIC Alpha size: ", mimic_alpha_results.shape)
        with open(config.outputs_results_dir + "mimic_alpha_%s.npy" % corpus_to_analysis, 'wb') as f:
            np.save(f, mimic_alpha_results)
        print("MIMIC Alpha (%s) saved" % corpus_to_analysis)


if __name__ == '__main__':

    # supermodel = AnnotationSuperModel()

    # supermodel.pretrain(base_step=0, is_train=True)
    # supermodel.pretrain(base_step=100000, is_train=False) # loss=1.9213, 1.819 (100 steps)
    # supermodel.pretrain(base_step=150000, is_train=False) # loss=2.1155, 1.655 (100 steps)
    # supermodel.pretrain(base_step=200000, is_train=False) # loss=1.7342, 1.534 (100 steps)
    # supermodel.embedding_space_analysis(pretrained_base_model=200000, corpus_to_analysis='hpo')
    # supermodel.embedding_space_analysis(pretrained_base_model=200000, corpus_to_analysis='test')
    # supermodel.embedding_space_analysis(pretrained_base_model=200000, corpus_to_analysis='train')
    # supermodel.annotation_training(base_step=0, pretrained_base_model=360000)

    unsuperised_controller = UnsupervisedAnnotationController()
    # unsuperised_controller.inference(base_step=75000, corpus_to_analysis='test')
    # unsuperised_controller.inference(base_step=75000, corpus_to_analysis='train')
    # unsuperised_controller.inference(base_step=10000, corpus_to_analysis='test')
    # unsuperised_controller.train(base_step=140000, is_train=True)
    # unsuperised_controller.train(base_step=1, is_train=True)

    # unsuperised_controller.inference(base_step=25000, corpus_to_analysis='test')
    # unsuperised_controller.inference(base_step=25000, corpus_to_analysis='train')

    # coef2 = 1
    # unsuperised_controller.train(base_step=0, is_train=True)
    # unsuperised_controller.inference(base_step=25000, corpus_to_analysis='test')
    # unsuperised_controller.inference(base_step=25000, corpus_to_analysis='train')

    # coef1 = 0.5, coef2 = 5, coef3 = 0.5
    # unsuperised_controller.train(base_step=40000, is_train=True)
    # unsuperised_controller.inference(base_step=55000, corpus_to_analysis='test')
    # unsuperised_controller.inference(base_step=55000, corpus_to_analysis='train')

    # unsuperised_controller.train(base_step=55000, is_train=True)
    # unsuperised_controller.inference(base_step=65000, corpus_to_analysis='test')
    # unsuperised_controller.inference(base_step=65000, corpus_to_analysis='train')

    # unsuperised_controller.train(base_step=155000, is_train=True)
    unsuperised_controller.inference(base_step=155000, corpus_to_analysis='test')
    # unsuperised_controller.inference(base_step=155000, corpus_to_analysis='train')

    # unsuperised_controller.train(base_step=170000, is_train=True)

    # TODO: CNN
    # unsuperised_controller.train(base_step=0, is_train=True)

    # TODO: mix hpo description with MIMIC
    # TODO: change transformer to CNN

    pass
