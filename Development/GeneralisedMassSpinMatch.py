#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew Lundgren, and Xan Morice-Atkinson 
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import os
import numpy as np
import pandas as pd
import time
import argparse
import logging

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from sklearn import model_selection

from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter

# Read command line option
parser = argparse.ArgumentParser(description=__doc__)

# Begin with code specific options
parser.add_argument("--neural-network-output-file", action="store", default=True,
                    help="Save the trained neural network model.")
parser.add_argument("-B", "--data", action="store", required=True,
                    help="The data required for the neural networkto be trained." 
                    "This will be split into training, validtaion and test datasets")

writer = SummaryWriter()
layout = {
    "ABCDE": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
        "accuracy": ["Multiline", ["accuracy/train", "accuracy/validation"]],
    },
}
writer.add_custom_scalars(layout)

#Defining functions
def to_cpu_np(x):
    return x.cpu().detach().numpy()

def scaling_the_mass(mass):
    scaled_mass = (mass - 51)/ np.sqrt(799)
    return scaled_mass

def rescaling_the_mass(mass):
    rescaled_mass = mass* np.sqrt(799) + 51
    return rescaled_mass

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#Upload the data
logging.info("Reading in the data")
TemplateBank = pd.read_csv(r'Unsorted1000000MassSpinTemplateBank.csv')

#Split the data
logging.info("Splitting the data into training, validtaion and test datasets")
TrainingBank, TestValidationBank = model_selection.train_test_split(TemplateBank, test_size=0.1)
TestBank, ValidationBank = model_selection.train_test_split(TestValidationBank, test_size=0.5)
logging.info(f'The data size of the training, validation and test dataset is {len(TrainingBank), len(ValidationBank), len(TestBank)}, respectively')

#Scaling the mass values and splitting into input (i.e. the template parameters) and output (the match)
logging.info("Scaling the mass values and splitting into input (i.e. the template parameters) and output (the match)")
x_train = np.vstack((scaling_the_mass(TrainingBank.ref_mass1.values), scaling_the_mass(TrainingBank.ref_mass2.values), 
scaling_the_mass(TrainingBank.mass1.values), scaling_the_mass(TrainingBank.mass2.values),
TrainingBank.ref_spin1.values, TrainingBank.ref_spin2.values,
TrainingBank.spin1.values, TrainingBank.spin2.values)).T
y_train = TrainingBank.match.values

x_val = np.vstack((scaling_the_mass(ValidationBank.ref_mass1.values), scaling_the_mass(ValidationBank.ref_mass2.values), 
scaling_the_mass(ValidationBank.mass1.values), scaling_the_mass(ValidationBank.mass2.values),
ValidationBank.ref_spin1.values, ValidationBank.ref_spin2.values,
ValidationBank.spin1.values, ValidationBank.spin2.values)).T
y_val = ValidationBank.match.values

x_test = np.vstack((scaling_the_mass(TestBank.ref_mass1.values), scaling_the_mass(TestBank.ref_mass2.values), 
scaling_the_mass(TestBank.mass1.values), scaling_the_mass(TestBank.mass2.values),
TestBank.ref_spin1.values, TestBank.ref_spin2.values,
TestBank.spin1.values, TestBank.spin2.values)).T
y_test = TestBank.match.values

logging.info("COnverting to datasets to a trainloader")
#Convert a numpy array to a Tensor
x_train = torch.tensor(x_train, dtype=torch.float32, device='cuda')
y_train = torch.tensor(y_train, dtype=torch.float32, device='cuda')
x_val = torch.tensor(x_val, dtype=torch.float32, device='cuda')
y_val = torch.tensor(y_val, dtype=torch.float32, device='cuda')
x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

#Convert data into trainloader
xy_train = TensorDataset(x_train, y_train)
xy_val = TensorDataset(x_val, y_val)
xy_test = TensorDataset(x_test, y_test)

training_loader  = DataLoader(xy_train, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(xy_val, batch_size=batch_size, drop_last=True)

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear0 = torch.nn.Linear(8, 341)
        self.linear1 = torch.nn.Linear(341, 65)
        self.linear2 = torch.nn.Linear(65, 342)
        self.linear3 = torch.nn.Linear(342, 74)
        self.linear_out = torch.nn.Linear(74, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear0(x))
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = self.linear_out(x)
        return x

start_time = time.time()
loss_list = []
val_list = []

epoch_number = 0
EPOCHS = 200
batch_size = 64

our_model = NeuralNetwork().to(device)
criterion = torch.nn.MSELoss(reduction='sum') # return the sum so we can calculate the mse of the epoch.
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min')

for epoch in range(EPOCHS):
    #print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    our_model.train(True)

    iters  = []
    epoch_loss = 0.
    val_iters = []
    vepoch_loss = 0.
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1)
        labels = labels.view(labels.size(0), -1)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = our_model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        del outputs
        del labels

        # Adjust learning weights
        optimizer.step()

        # save the current training information
        iters.append(i)
        batch_mse = loss/inputs.size(0)
        del batch_mse
        epoch_loss += float(loss) #added float to address the memory issues
    
    our_model.train(False)

    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.view(vinputs.size(0), -1)
        vlabels = vlabels.view(vlabels.size(0), -1)
        voutputs = our_model(vinputs)
        vloss = criterion(voutputs, vlabels)

        del voutputs
        del vlabels

        val_iters.append(i)
        vbatch_mse = vloss/vinputs.size(0)
        del vbatch_mse
        vepoch_loss += float(vloss)

    epoch_mse = epoch_loss/(len(iters) * inputs.size(0))
    vepoch_mse = vepoch_loss/(len(val_iters) * vinputs.size(0))
    print('EPOCH: {} LOSS train {} valid {}'.format(epoch_number, epoch_mse, vepoch_mse),end="\r")

    del epoch_loss
    del vepoch_loss
    del inputs
    del vinputs

    #if epoch%10==0: 
    #    torch.cuda.memory_summary(device=None, abbreviated=False)
    
    #This step to is determine whether the scheduler needs to be used 
    scheduler.step(vepoch_mse)

    #Create a loss_curve
    loss_list.append(epoch_mse)
    val_list.append(vepoch_mse)

    writer.add_scalar("loss/train", epoch_mse, epoch_number)
    writer.add_scalar("loss/validation", vepoch_mse, epoch_number)
    #writer.add_scalar("learning rate", scheduler._last_lr)
    epoch_number += 1

    del epoch_mse
    del vepoch_mse

print("Time taken", time.time() - start_time)
writer.flush()
writer.close()

#scheduler.optimizer.param_groups[0]['lr']
#optimizer.state

our_model.train(False)
#Time taken to predict the match on the test dataset  
with torch.no_grad():
    pred_start_time = time.time()
    y_prediction = our_model(x_test) 
    torch.cuda.synchronize()
    end_time = time.time()

print("Total time taken", end_time - pred_start_time)
print("Average time taken to predict the match", (end_time - pred_start_time)/len(x_test))

error = to_cpu_np(y_prediction[:, 0] - y_test)

#Determine the values it is failing
error_list = []
for x, y in zip(error, x_real):
    if abs(x) > 0.2:
        error_list.append(error)
        print(f'The error is {x} and the parameters are {y}')
    else:
        pass

#Save the Neural Network 
torch.save(our_model.state_dict(), '/users/sgreen/GPU/Width/saved_model_158.pth')

#A really complicated way of getting the list off the GPU and into a numpy array
loss_array = torch.tensor(loss_list, dtype=torch.float32, device='cpu').detach().numpy()
val_loss_array = torch.tensor(val_list, dtype=torch.float32, device='cpu').detach().numpy()

#Plots the loss curve for training and validation data set
plt.figure()
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F')
plt.semilogy(np.arange(1, len(val_loss_array)+1), val_loss_array, color='#0096FF')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('norm_loss_curve_158.pdf')

#Creates plots to compare the actual and predicted matches 
plt.figure()

x = to_cpu_np(y_test)
y = to_cpu_np(y_prediction[:, 0])
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.scatter(x,y,c=z, s=20) #, color='#5B2C6F')

plt.xlabel('Actual match')
plt.ylabel('Predicted match')
plt.savefig('norm_actual_predicted_158.pdf', dpi=300)

#Plots the distribution of errors for the test data
plt.figure()
plt.hist(error, bins=50, color='#5B2C6F')
plt.xlabel('Error')
plt.ylabel('Count')
plt.savefig('norm_error_158.pdf', dpi=300)

#Plots the distribution of errors for the test data where the actual match > .95
plt.figure()
plt.hist(error[x > .95], bins=50, color='#5B2C6F')
plt.xlabel('Error')
plt.ylabel('Count')
plt.savefig('norm_error_clipped_158.pdf', dpi=300)
