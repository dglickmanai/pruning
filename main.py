import os
import utils

# if on university
from args import get_args
from lib.data import get_loaders

isuni = os.path.isdir('/home/lab/glickmd1')
if isuni:
    os.environ["HF_DATASETS_CACHE"] = "/home/lab/glickmd1/.cache/huggingface/datasets"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.get_random_with_gpu_with_gb_free(60, 1))
# export HF_DATASETS_CACHE="/cortex/users/danielg/.cache/huggingface/datasets"
# export TRANSFORMERS_CACHE="/cortex/users/danielg/.cache/huggingface/transformers"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, prune_activations
from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if isuni else torch.float32,
        # cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        offload_folder="./offload" if not isuni else None,
    )

    model.seqlen = 2048
    return model


def main():
    args, dataloader, tokenizer = setup()

    pruning_experiment(args, dataloader, tokenizer)


def setup():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    return args, dataloader, tokenizer


def get_device_and_model(args):
    device = torch.device("cuda:0")
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    return device, model


def pruning_experiment(args, dataloader, tokenizer):
    if args.wandb_exp_name is not None and args.wandb_exp_name != "":
        import wandb
        wandb.init(project=args.wandb_exp_name, config=args)
        args = wandb.config

    device, model = get_device_and_model(args)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "activations":
            prune_activations(args, model, tokenizer, dataloader, device)
    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model, args) if not args.prune_method == "activations" else args.sparsity_ratio
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    ppl = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext {ppl}")
    wandb.log({"ppl": ppl, "sparsity": sparsity_ratio})
    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, "log.txt")
        with open(save_filepath, "w") as f:
            print("actual_sparsity\tppl", file=f, flush=True)
            print(f"{sparsity_ratio:.4f}\t{ppl:.4f}", file=f, flush=True)
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()
