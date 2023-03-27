#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

import torch
import torch.nn as nn

#Define the variables of the model 
INPUT = 8 #Number of inputs
LAYER1 = 341 #Number of neurons in the first layer
LAYER2 = 65 #Number of neurons in the second layer
LAYER3 = 342 #Number of neurons in the third layer
LAYER4 = 74 #Number of neurons in the fourth layer
OUTPUT = 1 #Number of outputs

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear0 = torch.nn.Linear(INPUT, LAYER1)
        self.linear1 = torch.nn.Linear(LAYER1, LAYER2)
        self.linear2 = torch.nn.Linear(LAYER2, LAYER3)
        self.linear3 = torch.nn.Linear(LAYER3, LAYER4)
        self.linear_out = torch.nn.Linear(LAYER4, OUTPUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear0(x))
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = self.linear_out(x)
        return x