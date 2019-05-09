
import os
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
        # self.config = models.CNNConfig(
        #     embed_hidden_size=512,
        #     max_position_embeddings=512,
        #     max_seq_len=32,
        #     num_hpo_node=len(dataloader.hpo_limited_list),
        #     hpo_hidden_size=512,
        #     vocab_size=len(self.tokenizer.vocab)
        # )

    def train(self, base_step=0, is_train=True):

        # create model for pretraining
        self.encoder = models.Encoder(self.config)
        # self.encoder = models.EncoderCNN(self.config)
        self.encoder.to(self.device)
        self.generator = models.Generator(self.config)
        self.generator.to(self.device)
        self.generator_prime = models.GeneratorPrime(self.config)
        self.generator_prime.to(self.device)

        if self.n_gpu > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.generator = torch.nn.DataParallel(self.generator)
            self.generator_prime = torch.nn.DataParallel(self.generator_prime)

        ## restore model if applicable
        if base_step > 0:
            logger.info("** ** * Restoring model from step %d ** ** * " % base_step)
            model_file = os.path.join(config.outputs_model_dir, "encoder_epoch%d.bin" % (base_step))
            state_dict = torch.load(model_file)
            self.encoder.load_state_dict(state_dict, strict=True)
            model_file = os.path.join(config.outputs_model_dir, "generator_epoch%d.bin" % (base_step))
            state_dict = torch.load(model_file)
            self.generator.load_state_dict(state_dict, strict=True)
            model_file = os.path.join(config.outputs_model_dir, "generator_prime_epoch%d.bin" % (base_step))
            state_dict = torch.load(model_file)
            self.generator_prime.load_state_dict(state_dict, strict=True)

        ## weights
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        encoder_weights = list(self.encoder.named_parameters())
        generator_weights = list(self.generator.named_parameters())
        generator_prime_weights = list(self.generator_prime.named_parameters())
        
        # res_weights = encoder_weights + generator_weights
        # pr_weights = encoder_weights + generator_prime_weights
        all_weights = encoder_weights + generator_weights + generator_prime_weights

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

        # TODO: which set to use?
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
        del hpo_children_info, corpus_hpo_data

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
        logger.info("  Num steps = %d", total_num_steps)

        if is_train:
            self.encoder.train()
            self.generator.train()
            self.generator_prime.train()
        else:
            self.encoder.eval()
            self.generator.eval()
            self.generator_prime.eval()

        hpo_ids = list()
        for hsample in hpo_root_dataset:
            hpo_ids.append(hsample[0])
        hpo_ids = torch.stack(hpo_ids, dim=0)
        assert hpo_ids.shape[0] == len(dataloader.hpo_limited_list)
        assert hpo_ids.shape[1] == config.sequence_length // 2

        for cur_epoch in range(total_num_epoch):

            def _hpo_res(coefficient1, coefficient2, coefficient3, percentage=1.0):

                res_hpo_loss_agg = res_alpha_loss_agg = prior_loss_agg = 0
                for step, batch in enumerate(hpo_corpus_dataloader):

                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_ids_hpos, input_mask, segment_ids, one_hpo_id, all_hpo_alpha = batch

                    #########
                    # reconstruction: encoding + generating
                    alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)
                    # full_reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, None)
                    full_reconstructed_sequence = self.generator(all_hpo_alpha, all_hpo_latent_outputs, None)

                    # torch.set_printoptions(threshold=5000)
                    # print(F.sigmoid(alpha_out)[:10])
                    # print(all_hpo_alpha[:10])
                    # exit()

                    res_hpo_loss = loss_function.resconstruction(full_reconstructed_sequence, input_ids)
                    res_alpha_loss = loss_function.alpha_cross_entropy(alpha_out, all_hpo_alpha)

                    #########
                    # prior constrain
                    one_hpo_latent_outputs = np.zeros(
                        shape=[all_hpo_latent_outputs.size()[0], all_hpo_latent_outputs.size()[2]]
                    ).astype(np.float32)
                    for i in range(one_hpo_id.shape[0]):
                        one_hpo_latent_outputs[i] = all_hpo_latent_outputs[i][one_hpo_id[i]].detach().cpu().numpy()
                    one_hpo_latent_outputs = torch.tensor(one_hpo_latent_outputs)
                    one_hpo_latent_outputs.to(self.device)

                    hpo_reconstructed_sequence = self.generator_prime(one_hpo_latent_outputs, None)
                    hpo_constrain_loss = loss_function.resconstruction(hpo_reconstructed_sequence, input_ids_hpos)

                    prior_loss = hpo_constrain_loss

                    loss = coefficient1 * res_hpo_loss + coefficient2 * res_alpha_loss + coefficient3 * prior_loss

                    if self.n_gpu > 1:
                        # res_loss = res_loss.mean() # mean() to average on multi-gpu.
                        res_hpo_loss = res_hpo_loss.mean()
                        res_alpha_loss = res_alpha_loss.mean()
                        prior_loss = prior_loss.mean()
                        loss = loss.mean()

                    res_hpo_loss_agg += res_hpo_loss.item()
                    res_alpha_loss_agg += res_alpha_loss.item()
                    prior_loss_agg += prior_loss.item()

                    # training the reconstruction part
                    if is_train:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # res_loss.backward()
                        # prior_loss.backward()
                        # optimizer_prime.step()
                        # optimizer_prime.zero_grad()

                    if step > 0 and step % 100 == 0:
                        logger.info("[HPO] E: %d, Step: %d, loss: %.4f / %.4f / %.4f" % (
                            cur_epoch, step, res_hpo_loss_agg / (step + 1), res_alpha_loss_agg / (step + 1),
                            prior_loss_agg / (step + 1)
                        ))

                    if step > len(hpo_corpus_dataloader) * percentage:
                        break

                if global_step % 100 == 0:
                    logger.info("[HPO] E: %d, Step: %d, loss: %.4f / %.4f / %.4f" % (
                        cur_epoch, step, res_hpo_loss_agg / (step + 1), res_alpha_loss_agg / (step + 1),
                        prior_loss_agg / (step + 1)
                    ))

            # training on HPO descriptions firstly
            if global_step + base_step == 0:
                for _ in range(5):
                    _hpo_res(0.01, 10, 0.01, percentage=1.0)

            # _hpo_res(0.01, 10, 0.01, percentage=1.0)
            # exit()

            tr_loss = 0
            tr_res_loss = 0
            tr_phe_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(mimic_corpus_dataloader, desc="MIMIC Iter")):

                ##########
                # reconstruction: encoding + generating
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids = batch
                # alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, input_mask)
                # full_reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, input_mask)
                alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)

                # torch.set_printoptions(threshold=5000)
                # tmp = F.softmax(alpha_out).unsqueeze(dim=-1)
                # print(tmp)
                # print(tmp.shape)
                # print(alpha_out.shape)
                # exit()

                full_reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, None)

                res_loss = loss_function.resconstruction(full_reconstructed_sequence, input_ids)

                ##########
                # prior constraint
                # randomly select hpos
                which_hpo = np.random.randint(low=0, high=len(dataloader.hpo_limited_list), size=(alpha_out.shape[0]))
                one_hpo_latent_outputs = np.zeros(
                    shape=[all_hpo_latent_outputs.size()[0], all_hpo_latent_outputs.size()[2]]
                ).astype(np.float32)
                hpo_input_ids = list()
                for i in range(which_hpo.shape[0]):
                    one_hpo_latent_outputs[i] = all_hpo_latent_outputs[i][which_hpo[i]].detach().cpu().numpy()
                    hpo_input_ids.append(hpo_ids[which_hpo[i]])
                one_hpo_latent_outputs = torch.tensor(one_hpo_latent_outputs)
                one_hpo_latent_outputs.to(self.device)
                hpo_input_ids = torch.stack(hpo_input_ids, dim=0)
                hpo_input_ids = hpo_input_ids.to(self.device)

                hpo_reconstructed_sequence = self.generator_prime(one_hpo_latent_outputs, None)

                pr_loss = loss_function.resconstruction(hpo_reconstructed_sequence, hpo_input_ids)

                loss = res_loss + pr_loss

                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    res_loss = res_loss.mean()
                    pr_loss = pr_loss.mean()

                tr_loss += loss.item()
                tr_res_loss += res_loss.item()
                tr_phe_loss += pr_loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if global_step % config.print_step_unsupervised == 0:
                    logger.info("[MIMIC] E: %d, Step: %d, Total Step: %d, loss: %.4f / %.4f" % (
                        cur_epoch, global_step, global_step + base_step,
                        tr_res_loss / nb_tr_steps, tr_phe_loss / nb_tr_steps))

                if is_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # res_loss.backward()
                    # pr_loss.backward()
                    # optimizer_prime.step()
                    # optimizer_prime.zero_grad()

                if global_step > 0 and global_step % 10 == 0:
                    _hpo_res(0.1, 1, 0.1,
                             percentage=0.02)

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
                    model_to_save = self.generator_prime
                    output_model_file = os.path.join(config.outputs_model_dir, "generator_prime_epoch%d.bin" %
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
        model_file = os.path.join(config.outputs_model_dir, "encoder_epoch%d.bin" % (base_step))
        state_dict = torch.load(model_file)
        self.encoder.load_state_dict(state_dict, strict=True)
        # model_file = os.path.join(config.outputs_model_dir, "generator_epoch%d.bin" % (base_step))
        # state_dict = torch.load(model_file)
        # self.generator.load_state_dict(state_dict, strict=True)

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

        loss_function = loss_func.Loss()

        ## start training
        logger.info("***** Running %s *****" % ('inference'))
        logger.info("  Num examples = %d", len(mimic_corpus_dataset))
        logger.info("  Batch size = %d", config.test_batch_size)
        logger.info("  Num steps = %d", total_num_steps)

        self.encoder.eval()

        # running on HPO descriptions firstly
        '''
        for step, batch in enumerate(hpo_corpus_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, alpha_ids = batch

            alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, input_mask)
            # alpha_out, all_hpo_latent_outputs = self.encoder(input_ids)
            reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, input_mask)
            # alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)
            # reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, None)

            res_loss = loss_function.resconstruction(reconstructed_sequence, input_ids)
            alpha_loss = loss_function.alpha_cross_entropy(alpha_out, alpha_ids)

            if self.n_gpu > 1:
                alpha_loss = alpha_loss.mean()
                res_loss = res_loss.mean()

            logger.info("[HPO] BaseStep: %d, loss: %.4f / %.4f" % (base_step, res_loss, alpha_loss))
        '''

        tr_loss = 0
        tr_res_loss = 0
        tr_phe_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        mimic_alpha_results = []
        for step, batch in enumerate(tqdm(mimic_corpus_dataloader, desc="MIMIC Iter")):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            # alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, input_mask)
            # reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, input_mask)
            alpha_out, all_hpo_latent_outputs = self.encoder(input_ids, segment_ids, None)
            # reconstructed_sequence = self.generator(alpha_out, all_hpo_latent_outputs, None)

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
    # unsuperised_controller.train(base_step=140000, is_train=True)
    unsuperised_controller.train(base_step=1, is_train=True)

    # TODO: mix hpo description with MIMIC
    # TODO: change transformer to CNN

    pass
