import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default='decapoda-research/llama-7b-hf')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"], default="unstructured")
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 'activations', "none"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true",
                        help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--activation_strength_metric', type=str, default='norm',
                        choices=['norm', 'var', 'percentile', 'reverse'])
    parser.add_argument('--weights_to_prune', metavar='N', type=str, nargs='*', default=[],
                        help='what weight matrixes to prune.'
                             'options are "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"'
                             'if none are specified, all are pruned')
    parser.add_argument('--ignore_init_masking_by_activations', action='store_true')
    parser.add_argument('--mask_train_epochs', type=int, default=0)
    parser.add_argument('--mask_train_lr', type=float, default=1e-3)
    parser.add_argument('--mask_train_bs', type=int, default=8)
    parser.add_argument('--mask_binarizer', type=str, default='binarize',
                        choices=['binarize', 'binarize_st', 'sigmoid_st', 'sigmoid'])
    parser.add_argument('--gradual_pruning', action='store_true')

    parser.add_argument('--wandb_exp_name', type=str, help='Wandb experiment name', default='pruning')
    args = parser.parse_args()
    return args
