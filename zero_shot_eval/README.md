# Zero-shot evaluation

Installation
```
cd lm-evaluation-harness
pip install -e . ## install lm-eval package which is able to take pruned models
```

Usage:
```
from evalutils import eval_llm 
results = eval_llm(model_name, model, tokenizer, task_list, num_shot, accelerate)
## task_list can be ["boolq","hellaswag","winogrande","arc_challenge","arc_easy", "openbookqa", "rte"]
## num_shot represents the number of few shot examples, 0 means zero-shot
## accelerate means whether the accelerate package is used for loading the model, a boolean variable;
```