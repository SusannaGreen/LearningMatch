#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

import torch
import torch.nn as nn

#Define the variables of the model 
INPUT = 4 #Number of inputs
OUTPUT = 1 #Number of outputs

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear0 = torch.nn.Linear(INPUT, 175)
        self.linear1 = torch.nn.Linear(175, 97)
        self.linear2 = torch.nn.Linear(97, 46)
        self.linear_out = torch.nn.Linear(46, OUTPUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear0(x))
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear_out(x)
        return x