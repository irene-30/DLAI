class SeqDataset(Dataset):
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]