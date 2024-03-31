import json
import os
import openai
import argparse
from tqdm import tqdm
from collections import defaultdict

from config import OPENAI_KEY

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-d", "--data", choices=["DC", "NS"], default="DC")
parser.add_argument("-p", "--prompt", type=str, default="")
args = parser.parse_args()

prompt_instruction = args.prompt.replace("\\n", "\n").strip("\n") # add \n in l.25
prompt4path = prompt_instruction.replace(' ', '_').replace('\n', '-')[:100]

if args.prompt:
    os.makedirs(f"results/{args.data}/{args.model}/{prompt4path}", exist_ok=True)

openai.api_key = OPENAI_KEY
article2tokens = json.load(open(f"data/{args.data}/tokens.json"))
article2surprisals = defaultdict(list)
for article_id, sents in article2tokens.items():
    print(article_id)
    for sent in tqdm(sents):
        prompt = prompt_instruction + "\n" + " ".join(sent)
        results = openai.Completion.create(
            model=args.model,
            prompt=prompt,
            echo=True,
            logprobs=1,
            max_tokens=0
        )
        surprisal_subwords = results["choices"][0]["logprobs"]["token_logprobs"]
        surprisal_offsets = results["choices"][0]["logprobs"]["text_offset"]
        # offset of targeted words #
        offset = []
        if args.prompt:
            i = len(prompt_instruction) + 1
        else:
            i = 1
        for t in sent:
            if i == 1 or (args.prompt and i == len(prompt_instruction) + 1):
                offset.append((i, i+len(t)-1))
                i += len(t) 
            else:
                offset.append((i, i+len(t)))
                i += len(t) + 1

        word_surprisals = []
        word_surprisal = []
        current_offset = offset.pop(0)
        for sup, i in zip(surprisal_subwords, surprisal_offsets):
            if i==0:
                continue
            elif args.prompt and i < len(prompt_instruction) + 1:
                continue
            else:
                if i <= current_offset[1]:
                    word_surprisal.append(sup)
                elif i > current_offset[1]:
                    word_surprisals.append(-sum(word_surprisal))
                    word_surprisal = [sup]
                    current_offset = offset.pop(0)
                    assert current_offset[0] == i
        if word_surprisal:
            word_surprisals.append(-sum(word_surprisal))
        article2surprisals[article_id].append(word_surprisals)

json.dump(article2surprisals, open(f"results/{args.data}/{args.model}/{prompt4path}/surprisal.json", "w"))
json.dump(prompt_instruction, open(f"results/{args.data}/{args.model}/{prompt4path}/prompt.json", "w"))