
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertLayerNorm, \
    BertEmbeddings, BertEncoder, BertPooler
from pytorch_pretrained_bert import BertModel, BertConfig

class BertForSentenceEmbedding(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSentenceEmbedding, self).__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        return _, pooled_output

class Config(object):
    def __init__(self,
                 hidden_size=256,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 intermediate_size=1024,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 max_seq_len=32,
                 num_hpo_node=24,
                 hpo_hidden_size=256,
                 vocab_size=30005, # 30000 words + 5 special tokens
    ):

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.max_seq_len = max_seq_len
        self.num_hpo_node = num_hpo_node
        self.hpo_hidden_size = hpo_hidden_size
        self.vocab_size = vocab_size


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        assert isinstance(config, Config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.hpo_dense = nn.Linear(config.hidden_size * config.max_seq_len, config.num_hpo_node)
        self.hpo_dense = nn.Linear(config.hidden_size, config.num_hpo_node)
        # self.hpo_activation = nn.ReLU()
        # self.hpo_softmax = nn.Softmax(dim=-1)
        # self.latent_dense_layers = nn.ModuleList([nn.Linear(config.hidden_size * config.max_seq_len, config.hpo_hidden_size) for _ in range(config.num_hpo_node)])
        self.latent_dense_layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hpo_hidden_size) for _ in range(config.num_hpo_node)])
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False)

        first_token_output = encoded_layers[-1][:, 0]
        alpha_output = self.hpo_dense(first_token_output)
        # alpha_output = self.hpo_softmax(self.hpo_activation(self.hpo_dense(first_token_output)))
        # all_token_output = encoded_layers[-1].view(-1, self.config.max_seq_len * self.config.hidden_size)
        # alpha_output = self.hpo_activation(self.hpo_dense(all_token_output))

        all_hpo_latent_outputs = []
        for layer in self.latent_dense_layers:
            hout = layer(first_token_output)
            all_hpo_latent_outputs.append(hout)
        # all_hpo_latent_outputs = []
        # for layer in self.latent_dense_layers:
        #     hout = layer(all_token_output)
        #     all_hpo_latent_outputs.append(hout)

        all_hpo_latent_outputs = torch.stack(all_hpo_latent_outputs, dim=1)

        return alpha_output, all_hpo_latent_outputs

class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.dense_latent = nn.Linear(config.hpo_hidden_size, config.max_seq_len * config.hidden_size)
        self.decoder = BertEncoder(config)
        self.dense = nn.Linear(config.hidden_size, config.vocab_size)
        # self.activation = nn.Softmax(dim=-1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, alpha_inputs, all_hpo_latent_inputs, attention_mask=None):
        assert alpha_inputs.shape[1] == all_hpo_latent_inputs.shape[1]
        latent_prime = torch.matmul(
            all_hpo_latent_inputs.permute(0, 2, 1),
            F.sigmoid(alpha_inputs).unsqueeze(dim=-1)
            # F.softmax(alpha_inputs).unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        hidden_state = self.dense_latent(latent_prime).view(-1, self.config.max_seq_len, self.config.hidden_size)

        if attention_mask is None:
            attention_mask = torch.ones(alpha_inputs.shape[0], self.config.max_seq_len)
            attention_mask = attention_mask.to(alpha_inputs.device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.decoder(hidden_state,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False)
        sequence_output = encoded_layers[-1]
        # sequence_vocab_output = self.activation(self.dense(sequence_output))
        sequence_vocab_output = self.dense(sequence_output)
        return sequence_vocab_output

class PriorConstraintModel(nn.Module):

    def __init__(self, config):
        super(PriorConstraintModel, self).__init__()
        self.config = config
        # self.dense_latent = nn.Linear(config.hpo_hidden_size, config.max_seq_len * config.hidden_size)
        # self.decoder = BertEncoder(config)
        # self.dense = nn.Linear(config.hidden_size, config.vocab_size)
        # self.activation = nn.Softmax(dim=-1)

        self.conv1 = nn.Conv1d(1, 4, 8, stride=4)
        self.conv2 = nn.Conv1d(4, 8, 4, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 2, stride=2)
        self.dense = nn.Linear(16 * 95, config.num_hpo_node)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, hpo_latent, attention_mask=None):

        z = hpo_latent.unsqueeze(dim=1)
        z = self.conv1(z)
        z = F.relu(z)
        z = self.conv2(z)
        z = F.relu(z)
        z = self.conv3(z)
        z = F.relu(z)
        z = z.view(z.shape[0], -1)
        z = self.dense(z)

        # hidden_state = self.dense_latent(hpo_latent).view(-1, self.config.max_seq_len, self.config.hidden_size)

        # if attention_mask is None:
        #     attention_mask = torch.ones(hpo_latent.shape[0], self.config.max_seq_len)
        #     attention_mask = attention_mask.to(hpo_latent.device)

        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # encoded_layers = self.decoder(hidden_state,
        #                               extended_attention_mask,
        #                               output_all_encoded_layers=False)
        # sequence_output = encoded_layers[-1]
        # sequence_vocab_output = self.activation(self.dense(sequence_output))
        # sequence_vocab_output = self.dense(sequence_output)
        # return sequence_vocab_output
        return z

class CNNConfig(object):
    def __init__(self,
        embed_hidden_size=256,
        max_position_embeddings=512,
        max_seq_len=32,
        num_hpo_node=24,
        hpo_hidden_size=512,
        vocab_size=30005, # 30000 words + 5 special tokens
    ):
        self.embed_hidden_size = embed_hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_seq_len
        self.num_hpo_node = num_hpo_node
        self.hpo_hidden_size = hpo_hidden_size
        self.vocab_size = vocab_size

class EncoderCNN(nn.Module):
    def __init__(self, config):
        super(EncoderCNN, self).__init__()
        assert isinstance(config, CNNConfig)
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embed_hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embed_hidden_size)
        self.conv1 = nn.Conv1d(config.embed_hidden_size, 512, 16, stride=1)
        self.max1 = nn.MaxPool1d(2, stride=1)
        self.conv2 = nn.Conv1d(512, 1024, 8, stride=1)
        self.max2 = nn.MaxPool1d(2, stride=1)
        self.conv3 = nn.Conv1d(1024, 2048, 4, stride=2)

        self.hpo_dense = nn.Linear(6144, config.num_hpo_node)
        # self.hpo_activation = nn.ReLU()
        # self.hpo_softmax = nn.Softmax(dim=-1)

        self.latent_dense_layers = nn.ModuleList([nn.Linear(6144, config.hpo_hidden_size) for _ in range(config.num_hpo_node)])
        # TODO: regularizers such as batchnorm dropout

    def forward(self, input_ids, _, __):
        # seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # words_embeddings = self.word_embeddings(input_ids)
        # position_embeddings = self.position_embeddings(position_ids)
        # embeddings = words_embeddings + position_embeddings
        embeddings = self.word_embeddings(input_ids)

        conv_out = self.conv1(embeddings.permute(0, 2, 1))
        conv_out = self.max1(F.relu(conv_out))
        conv_out = self.conv2(conv_out)
        conv_out = self.max2(F.relu(conv_out))
        conv_out = self.conv3(conv_out)
        conv_out = conv_out.view(-1, conv_out.shape[1] * conv_out.shape[2])

        alpha_out = self.hpo_dense(F.relu(conv_out))
        # alpha_out = self.hpo_activation(alpha_out)
        # alpha_out = self.hpo_softmax(alpha_out)

        all_hpo_latent_outputs = []
        for layer in self.latent_dense_layers:
            hout = layer(conv_out)
            all_hpo_latent_outputs.append(hout)

        all_hpo_latent_outputs = torch.stack(all_hpo_latent_outputs, dim=1)

        return alpha_out, all_hpo_latent_outputs

class GeneratorCNN(nn.Module):

    def __init__(self, config):
        super(GeneratorCNN, self).__init__()
        assert isinstance(config, CNNConfig)
        self.config = config

        self.conv1 = nn.ConvTranspose1d(config.hpo_hidden_size, 1024, 8, stride=2)
        self.max1 = nn.MaxPool1d(2, stride=1)
        self.conv2 = nn.ConvTranspose1d(1024, 2048, 4, stride=2)
        self.max2 = nn.MaxPool1d(2, stride=1)
        self.conv3 = nn.ConvTranspose1d(2048, 4096, 4, stride=2)
        # self.max3 = nn.MaxPool1d(2, stride=1)

        self.dense = nn.Linear(4096, self.config.vocab_size)

    def forward(self, alpha_inputs, all_hpo_latent_inputs, attention_mask=None):
        assert alpha_inputs.shape[1] == all_hpo_latent_inputs.shape[1]
        latent_prime = torch.matmul(
            all_hpo_latent_inputs.permute(0, 2, 1),
            F.sigmoid(alpha_inputs).unsqueeze(dim=-1)
            # F.softmax(alpha_inputs).unsqueeze(dim=-1)
        )

        z = self.conv1(latent_prime)
        z = F.relu(z)
        z = self.max1(z)
        z = self.conv2(z)
        z = F.relu(z)
        z = self.max2(z)
        z = self.conv3(z)
        z = F.relu(z)
        z = self.dense(z.permute(0, 2, 1))
        z = F.relu(z)

        return z


