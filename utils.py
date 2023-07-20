import os

import random


def get_gpu_memory(top=0):
    def top_values(d):
        # Sort the dictionary by values in descending order
        sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

        # Create a new dictionary containing only the first 2 items
        top_two_dict = dict(list(sorted_dict.items())[:top])

        return top_two_dict

    lines = os.popen('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free').readlines()

    memory_available = [int(x.split()[2]) for x in lines]
    gpus = {index: mb for index, mb in enumerate(memory_available)}
    if top:
        gpus = top_values(gpus)
    return gpus


def get_random_with_gpu_with_gb_free(gb, num_device=1):
    gpus = get_gpu_memory()
    filtered_keys = [str(key) for key, value in gpus.items() if value // 1000 > gb]
    if filtered_keys:
        return ','.join(random.sample(filtered_keys, num_device))
    else:
        raise ValueError(f"No GPU with {gb}GB available")
