import json
import os
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from config import HUFFINGFACE_KEY

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-q", "--quantize", default="")
parser.add_argument("-p", "--prompt", default="")
parser.add_argument("-d", "--prompt-data", default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
access_token = HUFFINGFACE_KEY
tokenizer = AutoTokenizer.from_pretrained(args.model, token=access_token)
tokenizer.pad_token = tokenizer.eos_token

if args.quantize == "4bit":
    gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, device_map="auto", load_in_4bit=True)
    gpt2_model.eval()
elif args.quantize == "8bit":
    gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, device_map="auto", load_in_8bit=True)
    gpt2_model.eval()
else:
    gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token)
    gpt2_model.to(device).eval() 

pipe = pipeline(task="text-generation", model=gpt2_model, tokenizer=tokenizer, device_map="auto")
os.makedirs(f"sampled/DC/{os.path.basename(args.model)}", exist_ok=True)

article2tokens = json.load(open("data/DC/tokens.json"))
if args.prompt_data:
    prompt_list = json.load(open(args.prompt_data))
else:
    prompt_list = [args.prompt]
for prompt in prompt_list:
    prompt = prompt.replace("\\n", "\n")
    prompt4path = prompt.replace(' ', '_').replace('\n', '-')[:100]
    texts = []
    for article_id, sents in article2tokens.items():
        print(article_id)
        results = pipe(prompt + " ".join(sents[1][:5]), max_new_tokens=50, do_sample=True, remove_invalid_values=True, top_p=0.95)
        text = results[0]["generated_text"]
        texts.append(text)
    json.dump(texts, open(f"sampled/DC/{os.path.basename(args.model)}/{prompt4path}.json", "w"))