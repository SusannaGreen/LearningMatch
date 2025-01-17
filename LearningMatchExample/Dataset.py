#!/usr/bin/env python

# Copyright (C) 2025 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

from typing import Iterator, List, Sized

import torch
from torch.utils.data.dataset import Tensor
from torch.utils.data.sampler import Sampler 
from torch.utils.data import TensorDataset

class MyDataset(TensorDataset):
    def __init__(self, *tensors: Tensor) -> None:
        super().__init__()
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitems__(self, indices):
        return tuple(tensor[indices] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

class FastRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized, generator=None) -> None:
        self.data_source = data_source
        self.generator = generator or torch.Generator()

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        rand_indices = torch.randperm(n, generator=self.generator).tolist()
        return iter(rand_indices)

    def __len__(self) -> int:
        return len(self.data_source)

class FastBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(iter(self.sampler))
        if self.drop_last:
            indices = indices[:len(indices) - len(indices) % self.batch_size]
        return (indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size