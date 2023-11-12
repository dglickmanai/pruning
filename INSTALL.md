# Installation  
Step 1: Create a new conda environment:
```
conda create -n prune_llm python=3.9
conda activate prune_llm
```
Step 2: Install relevant packages
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install evaluate
pip install scikit-learn
pip install transformers==4.35.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0

[//]: # (pip install bitsandbytes>=0.39.0)
[//]: # (pip install git+https://github.com/huggingface/accelerate.git)
```
There are known [issues](https://github.com/huggingface/transformers/issues/22222) with the transformers library on loading the LLaMA tokenizer correctly. Please follow the mentioned suggestions to resolve this issue.
