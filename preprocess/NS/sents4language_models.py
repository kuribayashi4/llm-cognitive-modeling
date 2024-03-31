import pandas as pd
import json
from statistics import mean
from collections import defaultdict

df = pd.read_table("data/NS/all.txt.annotation.filtered.csv")

article2sents = defaultdict(list)
for article_id, article_grouped in df.groupby("article"):
    for sent_id, sent_grouped in article_grouped.groupby("sent_id"):
        sent = list(sent_grouped["word"])
        article2sents[article_id].append(sent)
json.dump(article2sents, open("data/NS/tokens.json", "w"))

# def aggregate(x):
#     ts = [t for t in x if t > 0]
#     if ts:
#         return mean(ts)
#     else:
#         return 0

# avg_rt = pd.DataFrame(df.groupby(["article", "sent_id", "tokenN_in_sent"])["time"].apply(lambda x: aggregate(x)))
# assert len(avg_rt) == len([tok for article, sents in article2sents.items() for sent in sents for tok in sent])
# df_wo_rt = df.drop(["Unnamed: 0", "subj_id", "time", "logtime", "invtime", "id_in_sent"], axis=1).drop_duplicates()
# df_wo_rt = df_wo_rt.iloc[df_wo_rt[["article", "sent_id", "tokenN_in_sent"]].drop_duplicates().index]
# new_df = pd.merge(avg_rt, df_wo_rt, on=["article", "sent_id", "tokenN_in_sent"], how="left")
# new_df.to_csv("data/NS/all.txt.annotation.filtered.averaged_rt.csv")