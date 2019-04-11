
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert import BertModel

class BertForSentenceEmbedding(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSentenceEmbedding, self).__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        return pooled_output
