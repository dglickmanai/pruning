import wandb

from main import main

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_pruning',
    'metric': {'goal': 'maximize', 'name': 'ppl'},
    'parameters':
        {
            'prune_method': {'values': ['wanda', 'magnitude']},
            'sparsity_ratio': {'values': [0.1, 0.2, 0.3, 0.4, 0.5, ]},
        }
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='pruning'
)

wandb.agent(sweep_id, function=main)
