import time
import torch
import torch.nn as nn
from collections import defaultdict
import fnmatch
from lm_eval import tasks, evaluator 
import time 
import fnmatch

def eval_llm(model_name, model, tokenizer, task_list=["boolq","piqa","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], num_fewshot=0, use_accelerate=False):
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=/scratch/llm_weights"
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=/scratch/llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None, 
        # device='cuda:0',
        device=None,
        no_cache=True,
        # limit=None,
        limit=1000,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer 
    )

    ## results.keys(): ['results', 'versions', 'config']
    ## results["versions"]: {'lambada_standard': 0, 'lambada_openai_cloze': 0, 'lambada_openai': 0}
    ## results["results"]["lambada_standard"].keys(): dict_keys(['ppl', 'ppl_stderr', 'acc', 'acc_stderr'])
    return results 

def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def eval_lm_harness(model, tokenizer, task_list=None, num_fewshot=0):
    if task_list is None:
        task_names = tasks.ALL_TASKS 
    else:
        task_names = pattern_match(task_list, tasks.ALL_TASKS)

    results = evaluator.simple_evaluate(
        model="hf-causal",
        model_args="pretrained=decapoda-research/llama-7b-hf,cache_dir=/scratch/llm_weights",
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None, 
        device='cuda:0',
        no_cache=True,
        limit=None,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer 
    )

    return results 