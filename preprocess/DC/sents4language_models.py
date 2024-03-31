import pandas as pd
import json
from statistics import mean
from collections import defaultdict

df = pd.read_table("data/DC/all.txt.annotation.filtered.csv")
assert len(df) == 515010
article2sents = defaultdict(list)
for article_id, article_grouped in df.groupby("article"):
    c = 0
    for subj_id, subj_grouped in article_grouped.groupby("subj_id"):
        if c > 0:
            continue
        for sent_id, sent_grouped in subj_grouped.groupby("sent_id"):
            sent = []
            for tok_id, tok_grouped in sent_grouped.groupby("tokenN_in_sent"):
                tok = "".join(["".join(tok.replace("▁", " ").split()) for tok in tok_grouped["surface"]])
                sent.append(tok)
            article2sents[article_id].append(sent)
        c += 1
json.dump(article2sents, open("data/DC/tokens.json", "w"))

def aggregate(x):
    ts = [t for t in x if t > 0]
    if ts:
        return mean(ts)
    else:
        return 0

avg_rt = pd.DataFrame(df.groupby(["article", "sent_id", "tokenN_in_sent"])["time"].apply(lambda x: aggregate(x)))
assert len(avg_rt) == len([tok for article, sents in article2sents.items() for sent in sents for tok in sent])
df_wo_rt = df.drop(["Unnamed: 0", "subj_id", "time", "logtime", "invtime", "id_in_sent"], axis=1).drop_duplicates()
df_wo_rt = df_wo_rt.iloc[df_wo_rt[["article", "sent_id", "tokenN_in_sent"]].drop_duplicates().index]
new_df = pd.merge(avg_rt, df_wo_rt, on=["article", "sent_id", "tokenN_in_sent"], how="left")
new_df.to_csv("data/DC/all.txt.annotation.filtered.averaged_rt.csv")