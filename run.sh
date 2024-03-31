pip install -r requirements.txt

## DC preprocess
python preprocess/DC/add_annotation.py # some files are removed considering the dataset licence
python preprocess/DC/filter.py # zero filtering 
python preprocess/DC/sents4language_models.py

## NS preprocess
python preprocess/NS/filter.py
python preprocess/NS/sents4language_models.py


### preprocessed files are included in this repository, so oe can simply start the experiments below ####

## DC modeling
python experiments/calc_surprisal_gpt2.py -m gpt2 
python experiments/calc_surprisal_gpt2.py -m gpt2-medium 
python experiments/calc_surprisal_gpt2.py -m gpt2-large -b 2 -q 8bit
python experiments/calc_surprisal_gpt2.py -m gpt2-xl -b 2 -q 4bit

python experiments/calc_surprisal_gpt2.py -m facebook/opt-125m
python experiments/calc_surprisal_gpt2.py -m facebook/opt-350m
python experiments/calc_surprisal_gpt2.py -m facebook/opt-1.3b
python experiments/calc_surprisal_gpt2.py -m facebook/opt-2.7b
python experiments/calc_surprisal_gpt2.py -m facebook/opt-6.7b -q 8bit
python experiments/calc_surprisal_gpt2.py -m facebook/opt-13b -q 8bit
python experiments/calc_surprisal_gpt2.py -m facebook/opt-30b -q 8bit
python experiments/calc_surprisal_gpt2.py -m facebook/opt-66b -q 4bit

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -q 8bit
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -q 4bit
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-hf -b 2
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-hf -b 2 -q 8bit
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-hf -b 2 -q 4bit

python experiments/calc_surprisal_gpt2.py -b 2 -m tiiuae/falcon-7b-instruct
python experiments/calc_surprisal_gpt2.py -b 2 -q 4bit -m tiiuae/falcon-40b-instruct
python experiments/calc_surprisal_gpt2.py -b 2 -m tiiuae/falcon-7b
python experiments/calc_surprisal_gpt2.py -b 2 -q 4bit -m tiiuae/falcon-40b

python experiments/calc_surprisal_openai.py -m davinci-002 
python experiments/calc_surprisal_openai.py -m babbage-002 
python experiments/calc_surprisal_openai.py -m text-davinci-002 
python experiments/calc_surprisal_openai.py -m text-davinci-003 

## NS modeling
python experiments/calc_surprisal_gpt2.py -d NS -m gpt2 
python experiments/calc_surprisal_gpt2.py -d NS -m gpt2-d NS -medium 
python experiments/calc_surprisal_gpt2.py -d NS -m gpt2-large -b 2 -q 8bit
python experiments/calc_surprisal_gpt2.py -d NS -m gpt2-xl -b 2 -q 4bit

python experiments/calc_surprisal_gpt2.py -d NS -m facebook/opt-125m
python experiments/calc_surprisal_gpt2.py -d NS -m facebook/opt-350m
python experiments/calc_surprisal_gpt2.py -d NS -m facebook/opt-1.3b
python experiments/calc_surprisal_gpt2.py -d NS -m facebook/opt-2.7b
python experiments/calc_surprisal_gpt2.py -d NS -m facebook/opt-6.7b -q 8bit
python experiments/calc_surprisal_gpt2.py -d NS -m facebook/opt-13b -q 8bit
python experiments/calc_surprisal_gpt2.py -d NS -m facebook/opt-30b -q 8bit
python experiments/calc_surprisal_gpt2.py -d NS -m facebook/opt-66b -q 4bit

python experiments/calc_surprisal_llama.py -d NS -m meta-llama/Llama-2-7b-chat-hf -b 2
python experiments/calc_surprisal_llama.py -d NS -m meta-llama/Llama-2-13b-chat-hf -b 2 -q 8bit
python experiments/calc_surprisal_llama.py -d NS -m meta-llama/Llama-2-70b-chat-hf -b 2 -q 4bit
python experiments/calc_surprisal_llama.py -d NS -m meta-llama/Llama-2-7b-hf -b 2
python experiments/calc_surprisal_llama.py -d NS -m meta-llama/Llama-2-13b-hf -b 2 -q 8bit
python experiments/calc_surprisal_llama.py -d NS -m meta-llama/Llama-2-70b-hf -b 2 -q 4bit

python experiments/calc_surprisal_gpt2.py -b 2 -d NS -m tiiuae/falcon-7b-instruct
python experiments/calc_surprisal_gpt2.py -b 2 -q 4bit -d NS -m tiiuae/falcon-40b-instruct
python experiments/calc_surprisal_gpt2.py -b 2 -d NS -m tiiuae/falcon-7b
python experiments/calc_surprisal_gpt2.py -b 2 -q 4bit -d NS -m tiiuae/falcon-40b

python experiments/calc_surprisal_openai.py -d NS -m davinci-002 
python experiments/calc_surprisal_openai.py -d NS -m babbage-002 
python experiments/calc_surprisal_openai.py -d NS -m text-davinci-002 
python experiments/calc_surprisal_openai.py -d NS -m text-davinci-003 


## DC prompt
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a sentence."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a grammatically complex sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a grammatically simple sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a sentence using the most difficult vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a sentence using the simplest vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a sentence with a careful focus on grammar."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a sentence with a careful focus on word choice."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 --formatting -p "Please generate a sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a sentence."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a grammatically complex sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a grammatically simple sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a sentence using the most difficult vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a sentence using the simplest vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a sentence with a careful focus on grammar."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a sentence with a careful focus on word choice."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 --formatting -p "Please generate a sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a sentence."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a grammatically complex sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a grammatically simple sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a sentence using the most difficult vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a sentence using the simplest vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a sentence with a careful focus on grammar."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a sentence with a careful focus on word choice."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 --formatting -p "Please generate a sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."

python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

## DC regression
python experiments/dundee_modeling.py -i results/DC

## NS prompt
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a sentence."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a grammatically complex sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a grammatically simple sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a sentence using the most difficult vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a sentence using the simplest vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a sentence with a careful focus on grammar."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a sentence with a careful focus on word choice."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS --formatting -p "Please generate a sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a sentence."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a grammatically complex sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a grammatically simple sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a sentence using the most difficult vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a sentence using the simplest vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a sentence with a careful focus on grammar."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a sentence with a careful focus on word choice."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -d NS --formatting -p "Please generate a sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a sentence."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a grammatically complex sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a grammatically simple sentence as much as possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a sentence using the most difficult vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a sentence using the simplest vocabulary possible."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a sentence with a careful focus on grammar."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a sentence with a careful focus on word choice."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -d NS --formatting -p "Please generate a sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors."

python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-7b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_gpt2.py -m tiiuae/falcon-40b-instruct -b 2 -q 8bit -d NS -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-002 -d NS -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_openai.py -m text-davinci-003 -d NS -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

## NS regression
python experiments/dundee_modeling.py -i results/NS -d NS

## create metalinguistic prompts
experiments/create_prompt.ipynb

## DC metalinguistic prompt
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p data/NS/prompts/prompt_0.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p data/NS/prompts/prompt_1.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p data/NS/prompts/prompt_2.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-7b-chat-hf_0.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-7b-chat-hf_1.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-7b-chat-hf_2.txt -d DC -q 8bit

python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -p data/NS/prompts/prompt_0.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -p data/NS/prompts/prompt_1.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -p data/NS/prompts/prompt_2.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-13b-chat-hf_0.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-13b-chat-hf_1.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-13b-chat-hf_2.txt -d DC -q 8bit

python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -p data/NS/prompts/prompt_0.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -p data/NS/prompts/prompt_1.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -p data/NS/prompts/prompt_2.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-70b-chat-hf_0.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-70b-chat-hf_1.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -p data/NS/prompts/prompt_surprisal_Llama-2-70b-chat-hf_2.txt -d DC -q 4bit

python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -p data/NS/prompts/prompt_0.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -p data/NS/prompts/prompt_1.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -p data/NS/prompts/prompt_2.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -p data/NS/prompts/prompt_surprisal_falcon-7b-instruct_0.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -p data/NS/prompts/prompt_surprisal_falcon-7b-instruct_1.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -p data/NS/prompts/prompt_surprisal_falcon-7b-instruct_2.txt -d DC -q 8bit

python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -p data/NS/prompts/prompt_0.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -p data/NS/prompts/prompt_1.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -p data/NS/prompts/prompt_2.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -p data/NS/prompts/prompt_surprisal_falcon-40b-instruct_0.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -p data/NS/prompts/prompt_surprisal_falcon-40b-instruct_1.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -p data/NS/prompts/prompt_surprisal_falcon-40b-instruct_2.txt -d DC -q 4bit

python experiments/generate_openai.py -m text-davinci-002 -p data/NS/prompts/prompt_0.txt
python experiments/generate_openai.py -m text-davinci-002 -p data/NS/prompts/prompt_1.txt
python experiments/generate_openai.py -m text-davinci-002 -p data/NS/prompts/prompt_2.txt
python experiments/generate_openai.py -m text-davinci-002 -p data/NS/prompts/prompt_surprisal_text-davinci-002_0.txt
python experiments/generate_openai.py -m text-davinci-002 -p data/NS/prompts/prompt_surprisal_text-davinci-002_1.txt
python experiments/generate_openai.py -m text-davinci-002 -p data/NS/prompts/prompt_surprisal_text-davinci-002_2.txt

## NS metalinguistic prompt
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_0.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_1.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_2.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-7b-chat-hf_0.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-7b-chat-hf_1.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-7b-chat-hf_2.txt -d DC -q 8bit

python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_0.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_1.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_2.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-13b-chat-hf_0.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-13b-chat-hf_1.txt -d DC -q 8bit
python experiments/generate.py -m meta-llama/Llama-2-13b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-13b-chat-hf_2.txt -d DC -q 8bit

python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_0.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_1.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_2.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-70b-chat-hf_0.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-70b-chat-hf_1.txt -d DC -q 4bit
python experiments/generate.py -m meta-llama/Llama-2-70b-chat-hf -b 2 -d NS -p data/NS/prompts/prompt_surprisal_Llama-2-70b-chat-hf_2.txt -d DC -q 4bit

python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -d NS -p data/NS/prompts/prompt_0.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -d NS -p data/NS/prompts/prompt_1.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -d NS -p data/NS/prompts/prompt_2.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -d NS -p data/NS/prompts/prompt_surprisal_falcon-7b-instruct_0.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -d NS -p data/NS/prompts/prompt_surprisal_falcon-7b-instruct_1.txt -d DC -q 8bit
python experiments/generate.py -m tiiuae/falcon-7b-instruct -b 2 -d NS -p data/NS/prompts/prompt_surprisal_falcon-7b-instruct_2.txt -d DC -q 8bit

python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -d NS -p data/NS/prompts/prompt_0.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -d NS -p data/NS/prompts/prompt_1.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -d NS -p data/NS/prompts/prompt_2.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -d NS -p data/NS/prompts/prompt_surprisal_falcon-40b-instruct_0.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -d NS -p data/NS/prompts/prompt_surprisal_falcon-40b-instruct_1.txt -d DC -q 4bit
python experiments/generate.py -m tiiuae/falcon-40b-instruct -b 2 -d NS -p data/NS/prompts/prompt_surprisal_falcon-40b-instruct_2.txt -d DC -q 4bit

python experiments/generate_openai.py -m text-davinci-002 -d NS -p data/NS/prompts/prompt_0.txt
python experiments/generate_openai.py -m text-davinci-002 -d NS -p data/NS/prompts/prompt_1.txt
python experiments/generate_openai.py -m text-davinci-002 -d NS -p data/NS/prompts/prompt_2.txt
python experiments/generate_openai.py -m text-davinci-002 -d NS -p data/NS/prompts/prompt_surprisal_text-davinci-002_0.txt
python experiments/generate_openai.py -m text-davinci-002 -d NS -p data/NS/prompts/prompt_surprisal_text-davinci-002_1.txt
python experiments/generate_openai.py -m text-davinci-002 -d NS -p data/NS/prompts/prompt_surprisal_text-davinci-002_2.txt


## results of metalinguistic prompt
experiments/postporocesgeneartion.ipynb

## sampling texts ##
python experiments/sample_sents.py -m meta-llama/Llama-2-7b-chat-hf -d experiments/prompts4sample.json
python experiments/sample_sents.py -m meta-llama/Llama-2-13b-chat-hf -d experiments/prompts4sample.json -q 8bit
python experiments/sample_sents.py -m meta-llama/Llama-2-70b-chat-hf -d experiments/prompts4sample.json -q 4bit
python experiments/sample_sents.py -m tiiuae/falcon-7b-instruct -d experiments/prompts4sample.json -q 8bit
python experiments/sample_sents.py -m tiiuae/falcon-40b-instruct -d experiments/prompts4sample.json -q 4bit
python experiments/sample_sents.py -m tiiuae/falcon-40b-instruct -d experiments/prompts4sample.json -q 4bit

python experiments/sample_sents.py -m meta-llama/Llama-2-7b-chat-hf -d experiments/prompts4sample_old.json
python experiments/sample_sents.py -m meta-llama/Llama-2-13b-chat-hf -d experiments/prompts4sample_old.json -q 8bit
python experiments/sample_sents.py -m meta-llama/Llama-2-70b-chat-hf -d experiments/prompts4sample_old.json -q 4bit
python experiments/sample_sents.py -m tiiuae/falcon-7b-instruct -d experiments/prompts4sample_old.json -q 8bit
python experiments/sample_sents.py -m tiiuae/falcon-40b-instruct -d experiments/prompts4sample_old.json -q 4bit
python experiments/sample_sents.py -m tiiuae/falcon-40b-instruct -d experiments/prompts4sample_old.json -q 4bit

## analyze generated texts
visualization/generated_sents.ipynb

## analyze results
visualization/visualize.ipynb

## Appendix: LLama2 and format 1
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -b 2 -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -q 8bit -b 2 -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -q 4bit -b 2 -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"


python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-7b-chat-hf -d NS -b 2 -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-13b-chat-hf -d NS -q 8bit -b 2 -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"

python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence to make it as grammatically complex as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence to make it as grammatically simple as possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence using the most difficult vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence using the simplest vocabulary possible:\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence with a careful focus on grammar.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence with a careful focus on word choice.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence in a human-like manner. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"
python experiments/calc_surprisal_llama.py -m meta-llama/Llama-2-70b-chat-hf -d NS -q 4bit -b 2 -p "Please complete the following sentence. We are trying to reproduce human reading times with the word prediction probabilities you calculate, so please predict the next word like a human. It has been reported that human ability to predict next words is weaker than language models and that humans often make noisy predictions, such as careless grammatical errors.\\n"