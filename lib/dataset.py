import torch


def process_mprc(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                         # padding="max_length",
                         padding=True,
                         truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1"])
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence2"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


class MRPCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        encodings = tokenizer(dataset['sentence1'], dataset['sentence2'], truncation=True, padding="max_length")

        self.encodings = encodings
        self.labels = torch.tensor(dataset['label'])
        self.shape = self.labels.shape

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
