## Psychometric Predictive Power of Large Language Models
This repository contains the code of our NAACL 2024 (Findings) paper: [Psychometric Predictive Power of Large Language Models (Kuribayashi et al., 2024)](https://aclanthology.org/2024.findings-naacl.129/)  
This study explores how well large language models such as LLaMA-2 can simulate human reading behavior with their computed information-theoretic values.

## Experiments
First set `OPENAI_KEY` and `HUFFINGFACE_KEY` in `experiments/config.py`  with your API key.  
See `run.sh`

## Results
`results/{corpus}/{model}/{metric}.json`: PPL and PPP of base LLMs and IT-LLMs without any prompting  
`results/{corpus}/{model}/{prompt}/{metric}.json`: PPL and PPP of IT-LLMs with prompting  
`results/{corpus}/{model}/prompt_estimation/*.json`: output for metalinguistic prompting   
`sampled/DC/{model}/{prompt}.json`: generated text samples with prompts  
`visualization/visualize.ipynb`: source codes for generating figures and tables summarizing the results  

## Citation
```
@inproceedings{kuribayashi-etal-2024-psychometric,
    title = "Psychometric Predictive Power of Large Language Models",
    author = "Kuribayashi, Tatsuki  and
      Oseki, Yohei  and
      Baldwin, Timothy",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.129",
    pages = "1983--2005",
    abstract = "Instruction tuning aligns the response of large language models (LLMs) with human preferences.Despite such efforts in human{--}LLM alignment, we find that instruction tuning does not always make LLMs human-like from a cognitive modeling perspective. More specifically, next-word probabilities estimated by instruction-tuned LLMs are often worse at simulating human reading behavior than those estimated by base LLMs.In addition, we explore prompting methodologies for simulating human reading behavior with LLMs. Our results show that prompts reflecting a particular linguistic hypothesis improve psychometric predictive power, but are still inferior to small base models.These findings highlight that recent advancements in LLMs, i.e., instruction tuning and prompting, do not offer better estimates than direct probability measurements from base LLMs in cognitive modeling. In other words, pure next-word probability remains a strong predictor for human reading behavior, even in the age of LLMs.",
}
```
