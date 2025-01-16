#!/usr/bin/env python

# Copyright (C) 2025 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

import torch
import torch.nn as nn

#Define the Model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.embed = nn.Sequential(
              nn.Linear(4, 1024), 
              nn.ReLU(), 
              nn.Linear(1024, 1024),
              nn.ReLU(), 
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 16))
        self.crunch = nn.Sequential(
              nn.Linear(4+16, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.embed(x[..., 0:4])
        b = self.embed(x[..., 4:8])
        diffsq = torch.pow(torch.sub(a, b), 2)
        avg = torch.div(torch.add(x[..., 0:4],x[..., 4:8]), 2)
        return self.crunch(torch.concat((avg, diffsq), dim=1))