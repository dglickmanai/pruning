import torch


def process_mprc(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1"])
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence2"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets
