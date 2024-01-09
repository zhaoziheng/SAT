from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers import AutoModel,BertConfig, AutoTokenizer


class Text_Tower(nn.Module):
    """
    wrapper for a single text encoder, to allow text-text contrastive learning
    """
    def __init__(self,
                bert_model_name: str,
                embed_dim: int = 768):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.temperature, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)
    
    def _get_bert_basemodel(self, bert_model_name):
        config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        model = AutoModel.from_pretrained(bert_model_name, config=config)
        return model

    def lock(self):
        for param in self.parameters():
            param.requires_grad = False

    def encode_text(self, text):
        output = self.bert_model(input_ids = text['input_ids'],attention_mask = text['attention_mask'] )
        last_hidden_state, pooler_output, hidden_states = output[0],output[1],output[2]
        encode_out = self.mlp_embed(pooler_output)
        return encode_out
    
    def forward(self, text1, text2=None):
        text1_features = self.encode_text(text1)
        text1_features = F.normalize(text1_features, dim=-1)  # NOTE: normalized!
        if text2:
            text2_features = self.encode_text(text2)
            text2_features = F.normalize(text2_features, dim=-1)
        else:
            text2_features = None
        return text1_features, text2_features, self.temperature.exp()

    
