"""Tensor indexing equivalent to this study's single-process PyTorch DataLoader.

This is deliberately limited to TensorDataset, shuffle=True, replacement=False,
num_workers=0 and no dropped final batch. The runner checks batch order, generator
state, loss histories, probabilities and checkpoint records against DataLoader.
"""
import torch


class TensorLoader:
    def __init__(self, dataset, batch_size, shuffle, generator, num_workers):
        if not shuffle or num_workers != 0:
            raise ValueError('TensorLoader supports only the verified study settings.')
        self.tensors = dataset.tensors
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        # DataLoader draws a base seed before consuming RandomSampler's indices.
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        n = len(self.tensors[0])
        order = torch.randperm(n, generator=self.generator)
        for start in range(0, n, self.batch_size):
            indices = order[start:start + self.batch_size]
            yield tuple(tensor[indices] for tensor in self.tensors)
        # RandomSampler also draws a permutation for its zero-length remainder.
        torch.randperm(n, generator=self.generator)
