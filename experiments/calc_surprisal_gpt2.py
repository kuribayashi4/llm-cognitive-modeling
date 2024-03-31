import json
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

from config import HUFFINGFACE_KEY

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-b", "--batchsize", default=4, type=int)
parser.add_argument("-q", "--quantize", default="")
parser.add_argument("-p", "--prompt", type=str, default="")
parser.add_argument("-d", "--data", choices=["DC", "NS"], default="DC")
args = parser.parse_args()

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    article2tokens = json.load(open(f"data/{args.data}/tokens.json"))
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")

    access_token = HUFFINGFACE_KEY
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = args.prompt.replace("\\n", "\n")
    prompt4path = prompt.replace(' ', '_').replace('\n', '-')[:100]
    prompt_length = len(tokenizer(prompt, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"][0])

    os.makedirs(f"results/{args.data}/{os.path.basename(args.model)}", exist_ok=True)
    if args.prompt:
        os.makedirs(f"results/{args.data}/{os.path.basename(args.model)}/{prompt4path}", exist_ok=True)

    if args.quantize == "4bit":
        gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, device_map="auto", load_in_4bit=True)
        gpt2_model.eval()
    elif args.quantize == "8bit":
        gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, device_map="auto", load_in_8bit=True)
        gpt2_model.eval()
    else:
        gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token)
        gpt2_model.to(device).eval() 
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")

    bos_string = tokenizer.bos_token
    
    article2surprisals = defaultdict(list)
    article2entropies = defaultdict(list)
    article2renyi_entropies = defaultdict(list)

    for article_id, sents in article2tokens.items():
        print(article_id)
        for i in tqdm(range(0, len(sents), args.batchsize)):
            batch_sents = sents[i:i+args.batchsize]
            if prompt and prompt.endswith("\n"):
                tok_lss = []
                for sent in batch_sents:
                    tok_ls = []
                    for i, tok in enumerate(sent):
                        if i == 0:
                            tok_ls.append(len(tokenizer(tok, return_tensors="pt", add_special_tokens=False)["input_ids"][0]))
                        else:
                            tok_ls.append(len(tokenizer(" " + tok, return_tensors="pt", add_special_tokens=False)["input_ids"][0]))
                    tok_lss.append(tok_ls)
                encoded_sents = tokenizer([prompt + " ".join(sent) for sent in batch_sents], return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to(device)
                logits = gpt2_model(encoded_sents[:,:-1])[0][:,prompt_length-1:]
                target_ids = encoded_sents[:,prompt_length:]
            elif prompt and not prompt.endswith("\n"):
                tok_lss = []
                for sent in batch_sents:
                    tok_ls = [len(tokenizer(" " + tok, return_tensors="pt", add_special_tokens=False)["input_ids"][0]) for tok in sent]
                    tok_lss.append(tok_ls)
                encoded_sents = tokenizer([prompt + " " + " ".join(sent) for sent in batch_sents], return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to(device)
                logits = gpt2_model(encoded_sents[:,:-1])[0][:,prompt_length-1:]
                target_ids = encoded_sents[:,prompt_length:]
            else:
                tok_lss = []
                for sent in batch_sents:
                    tok_ls = [len(tokenizer(" " + tok, return_tensors="pt", add_special_tokens=False)["input_ids"][0]) for tok in sent]
                    tok_lss.append(tok_ls)
                encoded_sents = tokenizer([bos_string + " " + " ".join(sent) for sent in batch_sents], return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to(device)
                logits = gpt2_model(encoded_sents[:,:-1])[0]
                target_ids = encoded_sents[:,1:]
            surprisal_subwords = loss_fct(logits.transpose(1,2), target_ids)

            for sent_sup_sub, tok_ls in zip(surprisal_subwords, tok_lss):
                sent_sup_words = [sup.detach().sum().cpu().numpy().tolist() for _, sup in zip(tok_ls, torch.tensor_split(sent_sup_sub, torch.cumsum(torch.tensor(tok_ls), dim=0)))]
                article2surprisals[article_id].append(sent_sup_words)

            ps = logits.softmax(dim=-1)
            eps=1e-7

            entropy_subwords = (-ps*(ps+eps).log2()).sum(dim=-1)
            for sent_ent_sub, tok_ls in zip(entropy_subwords, tok_lss):
                sent_ent_words = [sup.detach().sum().cpu().numpy().tolist() for _, sup in zip(tok_ls, torch.tensor_split(sent_ent_sub, torch.cumsum(torch.tensor(tok_ls), dim=0)))]
                article2entropies[article_id].append(sent_ent_words)

            entropy_subwords = (ps**0.5).sum(dim=-1).log2() # renyi entropy alpha=1/2
            for sent_ent_sub, tok_ls in zip(entropy_subwords, tok_lss):
                sent_ent_words = [sup.detach().sum().cpu().numpy().tolist() for _, sup in zip(tok_ls, torch.tensor_split(sent_ent_sub, torch.cumsum(torch.tensor(tok_ls), dim=0)))]
                article2renyi_entropies[article_id].append(sent_ent_words)

            del logits
            del surprisal_subwords
            del entropy_subwords

        assert len(article2surprisals[article_id]) == len(sents)
        assert len([surprisal for sent_surprisals in article2surprisals[article_id] for surprisal in sent_surprisals]) == len([tok for sent in sents for tok in sent])

    if prompt:
        json.dump(article2surprisals, open(f"results/{args.data}/{os.path.basename(args.model)}/{prompt4path}/surprisal.json", "w"))
        json.dump(article2entropies, open(f"results/{args.data}/{os.path.basename(args.model)}/{prompt4path}/entropy.json", "w"))
        json.dump(article2renyi_entropies, open(f"results/{args.data}/{os.path.basename(args.model)}/{prompt4path}/renyi-entropy.json", "w"))
        json.dump(prompt, open(f"results/{args.data}/{os.path.basename(args.model)}/{prompt4path}/prompt.json", "w"))
    else:
        json.dump(article2surprisals, open(f"results/{args.data}/{os.path.basename(args.model)}/surprisal.json", "w"))
        json.dump(article2entropies, open(f"results/{args.data}/{os.path.basename(args.model)}/entropy.json", "w"))
        json.dump(article2renyi_entropies, open(f"results/{args.data}/{os.path.basename(args.model)}/renyi-entropy.json", "w"))

if __name__ == "__main__":
    main()