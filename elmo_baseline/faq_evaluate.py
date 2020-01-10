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
import pkuseg
from elmoformanylangs import Embedder

from metric import mean_reciprocal_rank

# 得到我们的elmo encoder
e = Embedder('/share/data/lang/users/zeweichu/projs/faqbot/zhs.model')
# sents = ["今天天气真好啊",
#         "潮水退了就知道谁没穿裤子"]

seg = pkuseg.pkuseg()
# sents = [seg.cut(sent) for sent in sents]

# print(sents)

# embeddings = e.sents2elmo(sents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="training file")
    parser.add_argument("--batch_size", default=64, type=int, required=False,
                        help="batch size for train and eval")
    args = parser.parse_args()

    # load dataset
    if not os.path.exists(args.train_file + "_embeddings.pkl"):
        train_df = pd.read_csv(args.train_file)
        candidates = train_df[train_df["is_best"] == 1][["title", "reply"]]
        candidate_title = candidates["title"].tolist()
        candidate_reply = candidates["reply"].tolist()

        titles = [seg.cut(title) for title in candidates["title"]]
        embeddings = e.sents2elmo(titles)
        # list of numpy arrays, each array with shape [seq_len * 1024]
        # code.interact(local=locals())
        candidate_embeddings = [np.mean(embedding, 0) for embedding in embeddings] # a list of 1024 dimensional vectors
        with open(args.train_file + "_embeddings.pkl", "wb") as fout:
            pickle.dump([candidate_title, candidate_reply, candidate_embeddings], fout)
    else:
        with open(args.train_file + "_embeddings.pkl", "rb") as fin:
            candidate_title, candidate_reply, candidate_embeddings = pickle.load(fin)

    df = pd.read_excel("../../dataset/faq/验证数据.xlsx", "投资知道")
    df = df[["问题", "匹配问题"]]
    df = df[df["匹配问题"].notna()]
    df = df[df["问题"].notna()]

    questions = df["问题"].tolist()
    matched_questions = df["匹配问题"].tolist()
    matched_questions_index = []
    for q in matched_questions:
        flg = False
        for i, _q in enumerate(candidate_title):
            if q == _q:
                matched_questions_index.append([i])
                flg = True
                break
        if flg == False:
            matched_questions_index.append([-1])
            
    matched_questions_index = np.asarray(matched_questions_index)
    questions = [seg.cut(q.strip()) for q in questions]
    question_embedding = [np.mean(emb, 0) for emb in e.sents2elmo(questions)] # 得到了新问题的ELMo embedding
    scores = cosine_similarity(question_embedding, candidate_embeddings)
    sorted_indices = scores.argsort()[:, ::-1]#[-5:][::-1]
    # code.interact(local=locals())
    mmr = mean_reciprocal_rank(sorted_indices == matched_questions_index)
    print("mean reciprocal rank: {}".format(mmr))

if __name__ == "__main__":
    main()
