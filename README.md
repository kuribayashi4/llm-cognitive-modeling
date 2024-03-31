## Psychometric Predictive Power of Large Language Models
This repository contains code of the paper: [Psychometric Predictive Power of Large Language Models (Kuribayashi et al., 2024)](https://arxiv.org/abs/2311.07484)  
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
This paper is accepted to Findings of NAACL 2024. If the official proceeding is provided, please cite it. 
```
@misc{kuribayashi2023psychometric,
      title={Psychometric Predictive Power of Large Language Models}, 
      author={Tatsuki Kuribayashi and Yohei Oseki and Timothy Baldwin},
      year={2023},
      eprint={2311.07484},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
