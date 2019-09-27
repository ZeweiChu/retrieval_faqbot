import argparse
from collections import Counter
import code
import os
import logging
from tqdm import tqdm, trange
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import AdamW, WarmupLinearSchedule
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import DataProcessor, InputExample

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

class FAQProcessor(DataProcessor):
	"""Processor for the FAQ problem"""
	def get_train_examples(self, data_dir):
		return self._create_examples(
			os.path.join(data_dir, "train.csv"))

	def get_dev_examples(self, data_dir):
		return self._create_examples(
			os.path.join(data_dir, "dev.csv"))
	
	def get_labels(self):
		return [0, 1]

	def _create_examples(self, path):
		df = pd.read_csv(path)
		examples = []
		titles = [str(t)[:100] for t in df["title"].tolist()]
		replies = [str(t)[:100] for t in df["reply"].tolist()]
		labels = df["is_best"].astype("int").tolist()
		for i in range(len(labels)):
			examples.append(
				InputExample(guid=i, text_a=titles[i], text_b=replies[i], label=labels[i]))
		return examples
		

def train(args, train_dataset, model, tokenizer):
	"""Train the model"""
	train_sampler = RandomSampler(train_dataset) 
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
	
	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)
	
	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
	set_seed(args)
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration")
		for step, batch in enumerate(epoch_iterator):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			inputs = {'input_ids':      batch[0],
					'attention_mask': batch[1],
					'token_type_ids': batch[2],
					'labels':         batch[3]}
			outputs = model(**inputs)
			loss = outputs[0]
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				scheduler.step()  # Update learning rate schedule
				optimizer.step()
				model.zero_grad()
				global_step += 1
			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

	return global_step, tr_loss / global_step				

def evaluate(args, model, tokenizer):
	pass
	# 请同学们自己完成这一部分


def load_and_cache_examples(args, tokenizer, evaluate=False):
	processor = FAQProcessor()
	cached_features_file = "cached_{}".format("dev" if evaluate else "train")	
	if os.path.exists(cached_features_file):
		features = torch.load(cached_features_file)
	else:
		label_list = processor.get_labels()
		examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
		features = convert_examples_to_features(
			examples=examples, 
			tokenizer=tokenizer, 
			max_length=args.max_seq_length, 
			label_list=label_list, 
			output_mode="classification", 
			pad_on_left=False,
			pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
			pad_token_segment_id=0)
		logger.info("Saving features into cached file %s", cached_features_file)
		torch.save(features, cached_features_file)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
	all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
	all_label = torch.tensor([f.label for f in features], dtype=torch.long)
	dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
	return dataset

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=42,
						help="random seed for initialization")
	parser.add_argument("--data_dir", default=None, type=str, required=True,
						help="directory containing the data")
	parser.add_argument("--output_dir", default="BERT_output", type=str, required=True,
						help="The model output save dir")
	parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
	parser.add_argument("--evaluate_during_training", action='store_true',
						help="Run evaluation during training at each logging step.")

	parser.add_argument("--max_seq_length", default=128, type=int, required=False, 
						help="maximum sequence length for BERT sequence classificatio")
	parser.add_argument("--max_steps", default=-1, type=int,
						help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	parser.add_argument("--warmup_steps", default=0, type=int,
						help="Linear warmup over warmup_steps.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--num_train_epochs", default=3.0, type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--learning_rate", default=5e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
						help="Weight deay if we apply some.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")

	parser.add_argument("--train_batch_size", default=64, type=int, required=False,
						help="batch size for train and eval")
	parser.add_argument('--logging_steps', type=int, default=50,
						help="Log every X updates steps.")
	
	args = parser.parse_args()
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt = '%m/%d/%Y %H:%M:%S',
						level = logging.INFO)

	# load dataset
	processor = FAQProcessor()
	tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="./cache")
	model = BertForSequenceClassification.from_pretrained('bert-base-chinese', cache_dir="./cache")
	
	label_list = processor.get_labels()
	num_labels = len(label_list)
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(args.device)
	# code.interact(local=locals())
	
	if args.do_train:
		train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
		global_step, tr_loss = train(args, train_dataset, model, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)

		logger.info("Saving model checkpoint to %s", args.output_dir)
		model_to_save = model.module if hasattr(model, 'module') else model
		model_to_save.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)
		torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
		model = BertForSequenceClassification.from_pretrained(args.output_dir)
		tokenizer = BertTokenizer.from_pretrained(args.output_dir)
		model = model.to(args.device)

if __name__ == "__main__":
	main()
