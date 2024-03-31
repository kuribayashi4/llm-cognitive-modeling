import json
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import pipeline

from experiments.config import HUFFINGFACE_KEY

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-b", "--batchsize", default=4, type=int)
parser.add_argument("-q", "--quantize", default="")
parser.add_argument("-p", "--prompt-file", type=str, required=True)
parser.add_argument("-d", "--data", choices=["DC", "NS"], default="DC")
args = parser.parse_args()


def create_prompt(prompt, sents):
    contexts = []
    for sent in sents:
        target = f'Suppose humans read the following sentence: "{" ".join(sent)}"\n'
        target += "List the tokens and their IDs in order of their reading cost (high to low) during sentence processing.\n"
        target += "Token ID:\n"
        target += " ".join([f"{str(i)}: {tok}," for i, tok in enumerate(sent)]) + "\n"
        target += "Answer:\n"
        target = prompt + target
        contexts.append(target)
    return contexts


def create_prompt_direct_rt(prompt, sents):
    contexts = []
    for sent in sents:
        target = f'Predict the reading time (ms) for each whitespace-separated word in the following sentence:\n{" ".join(sent)}\n'
        target = prompt + target
        contexts.append(target)
    return contexts


def create_prompt_surprisal(prompt, sents):
    contexts = []
    for sent in sents:
        target = f'Suppose you read the following sentence: "{" ".join(sent)}" \nList the tokens and their IDs in order of their probability in context (low to high).\n'
        target += "Token ID:\n"
        target += " ".join([f"{str(i)}: {tok}," for i, tok in enumerate(sent)]) + "\n"
        target += "Answer:\n"
        target = prompt + target
        contexts.append(target)
    return contexts


def create_prompt_direct_surprisal(prompt, sents):
    contexts = []
    for sent in sents:
        target = f'Predict the surprisal (bit) for each whitespace-separated word in the following sentence:\n{" ".join(sent)}\n'
        target = prompt + target
        contexts.append(target)
    return contexts

def get_prompt_func(prompt_file):
    if "surprisal" not in prompt_file and "direct" not in prompt_file:
        return create_prompt
    if "surprisal" in prompt_file and "direct" not in prompt_file:
        return create_prompt_surprisal
    elif "direct" in prompt_file and "surprisal" not in prompt_file:
        return create_prompt_direct_rt
    elif "direct" in prompt_file and "surprisal" in prompt_file:
        return create_prompt_direct_surprisal


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    article2tokens = json.load(open(f"data/{args.data}/tokens.json"))

    if args.data == "DC":
        article2tokens = {k: v for k, v in article2tokens.items() if int(k) < 6}

    access_token = HUFFINGFACE_KEY
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt_file = args.prompt_file
    prompt_func = get_prompt_func(prompt_file)
    with open(prompt_file) as f:
        prompt = "".join(f.readlines())

    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops = [], encounters=1):
            super().__init__()
            self.stops = [stop.to(device) for stop in stops]

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            last_token = input_ids[0][-1]
            for stop in self.stops:
                if tokenizer.decode(stop) == tokenizer.decode(last_token):
                    return True
            return False
        
    stop_words = ["\n", "\n\n"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    os.makedirs(f"results/{args.data}/{os.path.basename(args.model)}", exist_ok=True)
    os.makedirs(f"results/{args.data}/{os.path.basename(args.model)}/prompt_estimation", exist_ok=True)

    if args.quantize == "4bit":
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                    token=access_token, 
                                                    device_map="auto", 
                                                    load_in_4bit=True, 
                                                    bnb_4bit_compute_dtype=torch.float16)
        model.to_bettertransformer()
        model.eval()
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    elif args.quantize == "8bit":
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                                            token=access_token, 
                                            device_map="auto", 
                                            load_in_8bit=True)
        model.to_bettertransformer()
        model.eval()
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                                            token=access_token, 
                                            device_map="auto")
        model.to_bettertransformer()
        model.eval()
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    if "Llama" in args.model:
       tokenizer.pad_token = "[PAD]"
    
    article2output = defaultdict(list)
    for article_id, sents in article2tokens.items():
        for i in tqdm(range(0, len(sents), args.batchsize)):
            batch_sents = sents[i:i+args.batchsize]
            inputs = prompt_func(prompt, batch_sents)
            try:
                # (batchsize, 1, dict)
                outputs = pipe(inputs, max_new_tokens=500, do_sample=True, stopping_criteria=stopping_criteria, remove_invalid_values=True, top_p=0.95, batch_size=args.batchsize)
            except:
                outputs = pipe(inputs, max_new_tokens=500, do_sample=True, stopping_criteria=stopping_criteria, remove_invalid_values=True, top_p=0.95)
            for output in outputs:
                output_text = output[0]["generated_text"].strip()
                print(output_text)
                article2output[article_id].append(output_text)

    assert len(article2output[article_id]) == len(sents)

    json.dump(article2output, open(f"results/{args.data}/{os.path.basename(args.model)}/prompt_estimation/{os.path.basename(prompt_file)}.json", "w"))


if __name__ == "__main__":
    main()