import torch
from typing import Union, List
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MyTokenizer():
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
    
    def tokenize(self, texts:[str, List[str]]) -> torch.LongTensor:
        """
        tokenize a lits of strings or a single string, pad/trunctate to max length input

        Args:
            texts (str, List[str]]): a string 

        Returns:
            torch.LongTensor: the tokenized tensor and the attention mask(mask out paddings)
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = '[CLS]'
        eot_token = '[SEP]'
        all_token_ids = []
        max_len_in_this_batch = 0
        for text in texts:  # a string
            tokens = [sot_token] + self.tokenizer.tokenize(text) + [eot_token]  # list of str
            
            if len(tokens) > max_len_in_this_batch:
                max_len_in_this_batch = len(tokens)
            all_token_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))  
        if max_len_in_this_batch > self.max_length:
            max_len_in_this_batch = self.max_length
        result = torch.zeros(len(all_token_ids), max_len_in_this_batch, dtype=torch.long) 

        for i, token_ids in enumerate(all_token_ids):   # list of int
            if len(token_ids) > max_len_in_this_batch:
                token_ids = token_ids[:max_len_in_this_batch]  # Truncate
                token_ids[-1] = self.tokenizer.convert_tokens_to_ids('[SEP]')
            result[i, :len(token_ids)] = torch.tensor(token_ids)
            
        attn_mask = torch.where(result>0, 1, 0)
            
        return {'input_ids':result, 'attention_mask':attn_mask}