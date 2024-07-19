# -*- coding: UTF-8 -*-
from torch import nn
from transformers import BertModel

import params


class BertMultiClassification(nn.Module):

    def __init__(self, num_classes=40, dropout=0.3):
        super(BertMultiClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(params.MODEL_NAME, return_dict=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.dropout(x.pooler_output)
        x = self.linear(x)
        return x
