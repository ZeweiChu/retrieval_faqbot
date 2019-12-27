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
    if not os.path.exists("embeddings.pkl"):
        train_df = pd.read_csv(args.train_file)
        candidates = train_df[train_df["is_best"] == 1][["title", "reply"]]
        candidate_title = candidates["title"].tolist()
        candidate_reply = candidates["reply"].tolist()

        titles = [seg.cut(title) for title in candidates["title"]]
        embeddings = e.sents2elmo(titles)
        # code.interact(local=locals())
        candidate_embeddings = [np.mean(embedding, 0) for embedding in embeddings]
        with open("embeddings.pkl", "wb") as fout:
            pickle.dump([candidate_title, candidate_reply, embeddings], fout)
    else:
        with open("embeddings.pkl", "rb") as fin:
            candidate_title, candidate_reply, embeddings = pickle.load(fin)

    while True:
        title = input("你的问题是？\n")
        if len(title.strip()) == 0:
            continue
        title = [seg.cut(title.strip())] 
        title_embedding = [np.mean(e.sents2elmo(title)[0], 0)]
        scores = cosine_similarity(title_embedding, candidate_embeddings)[0]
        top5_indices = scores.argsort()[-5:][::-1]
        for index in top5_indices:
            print("可能的答案，参考问题：" + candidate_title[index] + "\t答案：" + candidate_reply[index] + "\t得分：" + str(scores[index]))

if __name__ == "__main__":
    main()
