# CUDA_VISIBLE_DEVICES=5 python main.py \
#     --model hf-causal \
#     --model_args pretrained=decapoda-research/llama-7b-hf,cache_dir=/scratch/llm_weights \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_challenge,arc_easy,openbookqa \
#     --num_fewshot 0 \
#     --device cuda:0


CUDA_VISIBLE_DEVICES=2 python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B,cache_dir=/scratch/llm_weights \
    --tasks lambada_openai,hellaswag \
    --device cuda:0