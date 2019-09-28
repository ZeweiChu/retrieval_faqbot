import torch

import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
	def __init__(self, 
		vocab_size, 
		embed_size, 
		hidden_size, 
		dropout_p=0.1, 
		avg_hidden=True,
		n_layers=1,
		bidirectional=True):
		super(GRUEncoder, self).__init__()
		self.hidden_size = hidden_size	
		self.embed = nn.Embedding(vocab_size, embed_size)
		if bidirectional:
			hidden_size //= 2
		self.rnn = nn.GRU(
			embed_size, 
			hidden_size,
			num_layers=n_layers,
			bidirectional=bidirectional,
			dropout=dropout_p)
		self.dropout = nn.Dropout(dropout_p)
		self.bidirectional = bidirectional
		self.avg_hidden = avg_hidden	

	def forward(self, x, mask):
		x_embed = self.embed(x)
		x_embed = self.dropout(x_embed)
		seq_len = mask.sum(1)
		# 下面部分是为了处理同一batch中不同长度的句子
		x_embed = torch.nn.utils.rnn.pack_padded_sequence(
			input=x_embed,
			lengths=seq_len,
			batch_first=True,
			enforce_sorted=False) 
		output, hidden = self.rnn(x_embed)
		output, seq_len = torch.nn.utils.rnn.pad_packed_sequence(
			sequence=output,
			batch_first=True,
			padding_value=0,
			total_length=mask.shape[1])
		if self.avg_hidden:
			hidden = torch.sum(output * mask.unsqueeze(2), 1) / torch.sum(mask, 1, keepdim=True)
		else:
			if self.bidirectional:
				hidden = torch.cat((hidden[-2,:,:], hidden[-1, :, :]), dim=1)
			else:
				hidden = hidden[-1, :, :]
		hidden = self.dropout(hidden)
		return hidden

class TransformerEncoder(nn.Module):
	pass



class DualEncoder(nn.Module):
	def __init__(self, encoder1, encoder2, type="cosine"):
		super(DualEncoder, self).__init__()
		self.encoder1 = encoder1
		self.encoder2 = encoder2
		if type == "CrossEntropy":
			self.linear = nn.Sequential(
				nn.Linear(self.encoder1.hidden_size + self.encoder2.hidden_size, 100),
				nn.ReLU(),
				nn.Linear(100, 1))

	def forward(self, x, x_mask, y, y_mask):
		x_rep = self.encoder1(x, x_mask)
		y_rep = self.encoder2(y, y_mask)
		return x_rep, y_rep

	def inference(self, x, x_mask, y, y_mask):
		x_rep, y_rep = self.forward(x, x_mask, y, y_mask)
		sim = F.cosine_similarity(x_rep, y_rep)
		return sim
