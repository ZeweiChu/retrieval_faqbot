import argparse
from collections import Counter
import code
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from models import GRUEncoder, DualEncoder

class Tokenizer():
	def __init__(self, vocab):
		self.id2word = ["UNK"] + vocab
		self.word2id = {w:i for i, w in enumerate(vocab)}

	def text2id(self, text):
		return [self.word2id.get(w, 0) for w in str(text)]

	def id2text(self, ids):
		return "".join([self.id2word[_id] for _id in ids])

	@property
	def vocab_size(self):
		return len(self.id2word)

def list2tensor(sents, tokenizer):
	res = []
	mask = []
	for sent in sents:
		res.append(tokenizer.text2id(sent))
	max_len = max([len(sent) for sent in res])
	for i in range(len(res)):
		res[i] = np.expand_dims(np.array(res[i] + [0] * (max_len - len(res[i]))), 0)
		_mask = np.zeros((1, max_len))
		_mask[:, :len(res[i])] = 1
		mask.append(_mask)
	res = np.concatenate(res, axis=0)
	mask = np.concatenate(mask, axis=0)
	res = torch.tensor(res).long()
	mask = torch.tensor(mask).float()
	return res, mask

def prepare_replies(df, model, device, tokenizer, args):
	model.eval()
	vectors = []
	for i in range(0, df.shape[0], args.batch_size):
		batch_df = df.iloc[i:i+args.batch_size]
		reply = list(batch_df["reply"])
		y, y_mask = list2tensor(reply, tokenizer)
		y = y.to(device)	
		y_mask = y_mask.to(device)
		y_rep = model.encoder2(y, y_mask)
		vectors.append(y_rep.data.cpu().numpy())
		
	replies = df["reply"].tolist()
	vectors = np.concatenate(vectors, 0)
	return replies, vectors 
		

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_file", default=None, type=str, required=True,
						help="training file")
	parser.add_argument("--output_dir", default=None, type=str, required=True,
						help="output directory for tokenizers and models")
	parser.add_argument("--batch_size", default=64, type=int, required=False,
						help="batch size for train and eval")
	parser.add_argument("--hidden_size", default=300, type=int, required=False,
						help="hidden size of GRU")
	parser.add_argument("--embed_size", default=300, type=int, required=False,
						help="word embedding size")
	args = parser.parse_args()

	# load dataset
	train_df = pd.read_csv(args.train_file)[["title", "reply"]]
	tokenizer = pickle.load(open(os.path.join(args.output_dir, "tokenizer.pickle"), "rb"))

	title_encoder = GRUEncoder(tokenizer.vocab_size, args.embed_size, args.hidden_size)
	reply_encoder = GRUEncoder(tokenizer.vocab_size, args.embed_size, args.hidden_size)
	model = DualEncoder(title_encoder, reply_encoder)
	model.load_state_dict(torch.load(os.path.join(args.output_dir, "faq_model.pth")))
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
	model = model.to(device)

	candidate_file = os.path.join(args.output_dir, "reply_candidates.pickle")
	if not os.path.isfile(candidate_file):
		replies, vectors = prepare_replies(train_df, model, device, tokenizer, args) 
		pickle.dump([replies, vectors], open(candidate_file, "wb"))
	else:
		replies, vectors = pickle.load(open(candidate_file, "rb"))
			

	while True:
		title = input("你的问题是？\n")
		if len(title.strip()) == 0:
			continue
		title = [title] 
		x, x_mask = list2tensor(title, tokenizer)
		x = x.to(device)	
		x_mask = x_mask.to(device)
		x_rep = model.encoder2(x, x_mask).data.cpu().numpy()
		scores = cosine_similarity(x_rep, vectors)[0]
		index = np.argmax(scores)
		print("可能的答案：", replies[index])

if __name__ == "__main__":
	main()
