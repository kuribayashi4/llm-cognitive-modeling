import json
import os
import argparse
import openai

from tqdm import tqdm
from collections import defaultdict

from config import OPENAI_KEY

openai.api_key = OPENAI_KEY

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-b", "--batchsize", default=4, type=int)
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

def main():
    article2tokens = json.load(open(f"data/{args.data}/tokens.json"))

    # if args.data == "DC":
        # article2tokens = {k: v for k, v in article2tokens.items() if int(k) < 6}
    
    prompt_file = args.prompt_file
    prompt_func = get_prompt_func(prompt_file)
    with open(prompt_file) as f:
        prompt = "".join(f.readlines())
        
    os.makedirs(f"results/{args.data}/{os.path.basename(args.model)}", exist_ok=True)
    os.makedirs(f"results/{args.data}/{os.path.basename(args.model)}/prompt_estimation", exist_ok=True)
    
    article2output = defaultdict(list)
    for article_id, sents in article2tokens.items():
        sents = sents[:5]
        print(article_id)
        for i in tqdm(range(0, len(sents), args.batchsize)):
            batch_sents = sents[i:i+args.batchsize]
            inputs = prompt_func(prompt, batch_sents)
            # print(inputs)
            results = openai.Completion.create(
                    model=args.model,
                    prompt=inputs,
                    temperature=0.0,
                    stop=["\n", "\n\n", "\n\n\n"],
                    max_tokens=200,
                )
            for i, output in enumerate(results["choices"]):
                output_text = output["text"].strip()
                print(output_text)
                article2output[article_id].append(inputs[i] + output_text)

    assert len(article2output[article_id]) == len(sents)
    json.dump(article2output, open(f"results/{args.data}/{os.path.basename(args.model)}/prompt_estimation/{os.path.basename(prompt_file)}.json", "w"))


if __name__ == "__main__":
    main()