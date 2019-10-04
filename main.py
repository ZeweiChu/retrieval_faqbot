import argparse
from collections import Counter
import code
import pickle
import os

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

def create_tokenizer(texts, vocab_size):
	"""
		args:
			texts: List[str] of text
	"""
	allvocab = ""
	for text in texts:
		allvocab += str(text)
	vocab_count = Counter(allvocab)
	vocab = vocab_count.most_common(vocab_size)
	vocab = [w[0] for w in vocab]
	return Tokenizer(vocab)

def list2tensor(sents, tokenizer):
	res = []
	mask = []
	for sent in sents:
		res.append(tokenizer.text2id(sent))
	max_len = max([len(sent) for sent in res])
	for i in range(len(res)):
		_mask = np.zeros((1, max_len))
		_mask[:, :len(res[i])] = 1
		res[i] = np.expand_dims(np.array(res[i] + [0] * (max_len - len(res[i]))), 0)
		mask.append(_mask)
	res = np.concatenate(res, axis=0)
	mask = np.concatenate(mask, axis=0)
	res = torch.tensor(res).long()
	mask = torch.tensor(mask).float()
	return res, mask

def train(df, model, loss_fn, optimizer, device, tokenizer, args):
	model.train()
	df = df.sample(frac=1)
	for i in range(0, df.shape[0], args.batch_size):
		batch_df = df.iloc[i:i+args.batch_size]
		title = list(batch_df["title"])
		reply = list(batch_df["reply"])
		target = torch.tensor(batch_df["is_best"].to_numpy()).float().view(-1, 1)
		if args.loss_function == "cosine":
			target[target==0] = -1
		x, x_mask = list2tensor(title, tokenizer)
		y, y_mask = list2tensor(reply, tokenizer)
	
		x = x.to(device)	
		x_mask = x_mask.to(device)
		y = y.to(device)	
		y_mask = y_mask.to(device)
		target = target.to(device)

		x_rep, y_rep = model(x, x_mask, y, y_mask)
		if args.loss_function == "cosine":
			loss = loss_fn(x_rep, y_rep, target)
		elif args.loss_function == "CrossEntropy":
			logits = model.linear(torch.cat([x_rep, y_rep], 1))
			loss = loss_fn(logits, target)
		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()

		if args.loss_function == "cosine":
			sim = F.cosine_similarity(x_rep, y_rep)
			sim[sim < 0] = -1
			sim[sim >= 0] = 1	
		elif args.loss_function == "CrossEntropy":
			sim = model.linear(torch.cat([x_rep, y_rep], 1))
			sim = torch.sigmoid(sim)
			sim[sim < 0.5] = 0
			sim[sim >= 0.5] = 1	

		sim = sim.view(-1)
		target = target.view(-1)
		acc = torch.sum(sim == target).item() / target.shape[0]
		if i % 100 == 0:
			print("iteration: {}, loss: {}, accuracy: {}".format(i, loss.item(), acc))

def evaluate(df, model, loss_fn, device, tokenizer, args):
	model.eval()
	df = df.sample(frac=1)
	num_corrects, total_counts = 0, 0
	for i in range(0, df.shape[0], args.batch_size):
		batch_df = df.iloc[i:i+args.batch_size]
		title = list(batch_df["title"])
		reply = list(batch_df["reply"])
		target = torch.tensor(batch_df["is_best"].to_numpy()).float().view(-1,1)
		if args.loss_function == "cosine":
			target[target==0] = -1
		x, x_mask = list2tensor(title, tokenizer)
		y, y_mask = list2tensor(reply, tokenizer)
	
		x = x.to(device)	
		x_mask = x_mask.to(device)
		y = y.to(device)	
		y_mask = y_mask.to(device)
		target = target.to(device)

		x_rep, y_rep = model(x, x_mask, y, y_mask)
		if args.loss_function == "cosine":
			loss = loss_fn(x_rep, y_rep, target)
			sim = F.cosine_similarity(x_rep, y_rep)
			sim[sim < 0] = -1
			sim[sim >= 0] = 1	
		elif args.loss_function == "CrossEntropy":
			logits = model.linear(torch.cat([x_rep, y_rep], 1))
			loss = loss_fn(logits, target)
			sim = torch.sigmoid(logits)
			sim[sim < 0.5] = 0
			sim[sim >= 0.5] = 1	

		sim = sim.view(-1)
		target = target.view(-1)
		num_corrects = torch.sum(sim == target).item() 
		total_counts = target.shape[0]
	
	print("accuracy: {}".format(num_corrects / total_counts))
	return num_corrects / total_counts
		

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_file", default=None, type=str, required=True,
						help="training file")
	parser.add_argument("--dev_file", default=None, type=str, required=True,
						help="development file")
	parser.add_argument("--output_dir", default=None, type=str, required=True,
						help="output directory for tokenizers and models")
	parser.add_argument("--num_epochs", default=10, type=int, required=False,
						help="number of epochs for training")
	parser.add_argument("--vocab_size", default=50000, type=int, required=False,
						help="vocabulary size")
	parser.add_argument("--hidden_size", default=300, type=int, required=False,
						help="hidden size of GRU")
	parser.add_argument("--embed_size", default=300, type=int, required=False,
						help="word embedding size")
	parser.add_argument("--batch_size", default=64, type=int, required=False,
						help="batch size for train and eval")
	parser.add_argument("--loss_function", default="CrossEntropy", type=str, required=False,
						choices=["CrossEntropy", "cosine"],
						help="which loss function to choose")
	args = parser.parse_args()

	# load dataset
	train_df = pd.read_csv(args.train_file)[["title", "reply", "is_best"]]
	dev_df = pd.read_csv(args.dev_file)[["title", "reply", "is_best"]]
	texts = list(train_df["title"]) + list(train_df["reply"])
	tokenizer = create_tokenizer(texts, args.vocab_size)

	title_encoder = GRUEncoder(tokenizer.vocab_size, args.embed_size, args.hidden_size)
	reply_encoder = GRUEncoder(tokenizer.vocab_size, args.embed_size, args.hidden_size)
	model = DualEncoder(title_encoder, reply_encoder, type=args.loss_function)
	if args.loss_function == "CrossEntropy":
		loss_fn = nn.BCEWithLogitsLoss()
	elif args.loss_function == "cosine":
		loss_fn = nn.CosineEmbeddingLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
	model = model.to(device)
	
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	pickle.dump(tokenizer, open(os.path.join(args.output_dir, "tokenizer.pickle"), "wb"))

	best_acc = 0.
	for epoch in range(args.num_epochs):
		print("start epoch {}".format(epoch))
		train(train_df, model, loss_fn, optimizer, device, tokenizer, args)	
		acc = evaluate(dev_df, model, loss_fn, device, tokenizer, args)
		if acc > best_acc:
			best_acc = acc
			print("saving best model")
			torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))

if __name__ == "__main__":
	main()
