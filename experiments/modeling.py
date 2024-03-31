import json
import os
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse
from statistics import mean
from collections import defaultdict
from patsy import dmatrices

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", required=True)
parser.add_argument("-d", "--data", choices=["DC", "NS"], default="DC")
parser.add_argument("--interaction", action="store_true")
args = parser.parse_args()

df_original = pd.read_csv(f"data/{args.data}/all.txt.annotation.filtered.averaged_rt.csv")
df_original = df_original.sort_values(["article", "sent_id", "tokenN_in_sent"])

for metrics in ["surprisal", "entropy", "renyi-entropy"]:
    print(metrics)
    target_files = glob.glob(f"{args.input_dir}/**/{metrics}.json", recursive=True)
    if not target_files:
        continue
    for target_file in target_files:
        print(target_file)
        article2interest = json.load(open(target_file))
        all_interests = [sup for _, sents_sups in sorted(article2interest.items(), key=lambda x: int(x[0])) for sent_sups in sents_sups for sup in sent_sups]
        assert len(df_original) == len(all_interests)
        df = df_original.copy()
        df["interest"] = all_interests
        
        context_surprisal_file = os.path.join(os.path.dirname(target_file), "surprisal.json")
        article2surprisals = json.load(open(target_file))
        all_sups = [sup for _, sents_sups in sorted(article2surprisals.items(), key=lambda x: int(x[0])) for sent_sups in sents_sups for sup in sent_sups]
        mean_surprisal = mean(all_sups)
        assert len(df_original) == len(all_sups)

        article2prev_surprisals = defaultdict(list)
        article2prev_prev_surprisals = defaultdict(list)
        article2prev_prev_prev_surprisals = defaultdict(list)
        for article_id, sents_sups in article2surprisals.items():
            for sent_sups in sents_sups:
                article2prev_surprisals[article_id].extend([mean_surprisal] + sent_sups[:-1])
                if len(sent_sups) == 1:
                    article2prev_prev_surprisals[article_id].extend([mean_surprisal])
                    article2prev_prev_prev_surprisals[article_id].extend([mean_surprisal])
                elif len(sent_sups) == 2:
                    article2prev_prev_surprisals[article_id].extend([mean_surprisal, mean_surprisal])
                    article2prev_prev_prev_surprisals[article_id].extend([mean_surprisal, mean_surprisal])
                else:
                    article2prev_prev_surprisals[article_id].extend([mean_surprisal, mean_surprisal] + sent_sups[:-2])
                    article2prev_prev_prev_surprisals[article_id].extend([mean_surprisal, mean_surprisal, mean_surprisal] + sent_sups[:-3])

        df["surprisal_prev_1"] = [sup for article_id, sups in sorted(article2prev_surprisals.items(), key=lambda x: x[0]) for sup in sups]
        df["surprisal_prev_2"] = [sup for article_id, sups in sorted(article2prev_prev_surprisals.items(), key=lambda x: x[0]) for sup in sups]
        df["surprisal_prev_3"] = [sup for article_id, sups in sorted(article2prev_prev_prev_surprisals.items(), key=lambda x: x[0]) for sup in sups]

        df = df[df["time"] > 0]
        df = df[df["tokenN_in_sent"] > 0]
        df = df[df["is_first"] == False]
        df = df[df["is_last"] == False]


        if args.interaction:
            y, X = dmatrices('time ~ interest + surprisal_prev_1 + surprisal_prev_2 + length*log_gmean_freq + length_prev_1*log_gmean_freq_prev_1 + length_prev_2*log_gmean_freq_prev_2', data=df, return_type='dataframe')
            mod = sm.OLS(y, X)
            res = mod.fit() 

            y_baseline, X_baseline = dmatrices('time ~  surprisal_prev_1 + surprisal_prev_2 + length*log_gmean_freq  + length_prev_1*log_gmean_freq_prev_1 + length_prev_2*log_gmean_freq_prev_2', data=df, return_type='dataframe')
            mod_baseline = sm.OLS(y_baseline, X_baseline)
            res_baseline = mod_baseline.fit() 
            with open(target_file + ".interaction.result", "w") as f:
                f.write(f"delta loglik: {res.llf - res_baseline.llf}\n")
                f.write(f"delta loglik per tokens: {(res.llf - res_baseline.llf)/len(df)}\n")
                f.write(f"average surprisal: {df['interest'].mean()/np.log(2)}\n")
                f.write(f"perplexity: {np.exp(df['interest'].mean())}\n")
                f.write(str(res.summary()))
        else:
            y, X = dmatrices('time ~ interest + surprisal_prev_1 + surprisal_prev_2 + length + log_gmean_freq + length_prev_1 + log_gmean_freq_prev_1 + length_prev_2 + log_gmean_freq_prev_2', data=df, return_type='dataframe')
            mod = sm.OLS(y, X)
            res = mod.fit() 

            y_baseline, X_baseline = dmatrices('time ~  surprisal_prev_1 + surprisal_prev_2 + length + log_gmean_freq  + length_prev_1 + log_gmean_freq_prev_1 + length_prev_2 + log_gmean_freq_prev_2', data=df, return_type='dataframe')
            mod_baseline = sm.OLS(y_baseline, X_baseline)
            res_baseline = mod_baseline.fit() 
            with open(target_file + ".result", "w") as f:
                f.write(f"delta loglik: {res.llf - res_baseline.llf}\n")
                f.write(f"delta loglik per tokens: {(res.llf - res_baseline.llf)/len(df)}\n")
                f.write(f"average surprisal: {df['interest'].mean()/np.log(2)}\n")
                f.write(f"perplexity: {np.exp(df['interest'].mean())}\n")
                f.write(str(res.summary()))