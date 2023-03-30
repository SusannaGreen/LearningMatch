#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

import torch
import torch.nn as nn

#Define the variables of the model 
INPUT = 8 #Number of inputs
OUTPUT = 1 #Number of outputs

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear0 = torch.nn.Linear(INPUT, 341)
        self.linear1 = torch.nn.Linear(341, 75)
        self.linear2 = torch.nn.Linear(75, 341)
        self.linear3 = torch.nn.Linear(341, 74)
        self.linear_out = torch.nn.Linear(74, OUTPUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear0(x))
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = self.linear_out(x)
        return x