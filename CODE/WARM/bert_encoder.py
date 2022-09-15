import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_transformers import BertModel

class BertEncoder(nn.Module):
	def __init__(self, args, model):
		super(BertEncoder,self).__init__()
		self.args = args
		self.device = None
		self.bert = BertModel.from_pretrained(model)
		self.lin = nn.Linear(768, args.hidden_size)
	
	def forward(self, input_ids, token_type_ids, attention_mask):
		outputs = self.bert(input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda())
		outputs = self.lin(outputs[1])
		return outputs.unsqueeze(0)
	
	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0]
		self.bert = self.bert.to(*args, **kwargs)
		return self