import wandb

from main import main, setup, pruning_experiment

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_pruning',
    'metric': {'goal': 'maximize', 'name': 'ppl'},
    'parameters':
        {
            'prune_method': {'values': ['activations']},
            'weights_to_prune': {
                # 'values': [[], ['q_proj'], ['k_proj'], ['v_proj'], ['o_proj'], ['gate_proj'], ['down_proj'],
                #            ['up_proj']]
                'values': [['q_proj', 'k_proj', 'down_proj']]
            },
            'sparsity_ratio': {
                'values': [0.1, 0.2, 0.3, 0.5, ]
                # 'values': [0.1, 0.2, 0.3, 0.4]
            },
        }
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='pruning'
)
# sweep_id = 'daniel-ai/pruning/3ixbddfg'

args, dataloader, tokenizer = setup()


def experiment():
    pruning_experiment(args, dataloader, tokenizer)


wandb.agent(sweep_id, function=experiment)
