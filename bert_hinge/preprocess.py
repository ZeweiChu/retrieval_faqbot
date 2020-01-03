import pandas as pd
import code

df = pd.read_csv("../../dataset/faq/lawzhidao_filter.csv")
df = df[df["is_best"] == 1]

translated = pd.read_csv("synonym.tsv", sep="\t", header=0)


merged = df.merge(translated, left_on="title", right_on="original")
merged[["title", "reply", "translated"]].to_csv("preprocessed.csv", index=False, sep="\t")
