import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .tokenizer import MyTokenizer


class Text_Tower(nn.Module):
    def __init__(self, biolord_checkpoint: str = None,):
        super().__init__()

        self.biolord = AutoModel.from_pretrained(biolord_checkpoint)
        self.tokenizer = MyTokenizer(biolord_checkpoint, 256)
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text):
        text = self.tokenizer.tokenize(text) # (n, max_l)
        text['input_ids'] = text['input_ids'].to(device=torch.cuda.current_device())
        text['attention_mask'] = text['attention_mask'].to(device=torch.cuda.current_device())
            
        output = self.biolord(**text)
        pooler_output = self.mean_pooling(output, text['attention_mask'])
        
        return pooler_output

    
