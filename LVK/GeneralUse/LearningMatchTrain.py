#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew Lundgren, and Xan Morice-Atkinson 

from LearningMatchModel.py import NeuralNetwork

import os
import numpy as np
import pandas as pd
import time
import argparse
import logging

from sklearn import model_selection

from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

#Define where to find the relavent datasets
TRAINING_BANK = r'/users/sgreen/LearningMatch/Development/MassSpinTrainingBank.csv'
VALIDATION_BANK = r'/users/sgreen/LearningMatch/Development/MassSpinValidationBank.csv'

#Define values for the training of the model
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

#Define where to save the trained model and other outputs for the LearningMatchplots.py
LEARNINGMATCH_MODEL = '/users/sgreen/LearningMatch/Development/LearningMatchModel.pth'
TRAINING_LOSS = '/users/sgreen/LearningMatch/Development/TrainingLoss.csv'
VALIDATION_LOSS = '/users/sgreen/LearningMatch/Development/ValidationLoss.csv'

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#Reading the Training, validation and test dataset
logging.info("Reading in the data")
TrainingBank = pd.read_csv(TRAINING_BANK)
ValidationBank = pd.read_csv(VALIDATION_BANK)
logging.info(f'The data size of the training and validation dataset is {len(TrainingBank), len(ValidationBank)}, respectively')

#Using Standard.scalar() to re-scale the Training Bank and applying to the validation and test bank. 
scaler = preprocessing.StandardScaler()
TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.fit_transform(TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])
ValidationBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.transform(ValidationBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])
TestBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.transform(TestBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])
scaler_mean = scaler.mean_
scaler_std = scaler.std_
logging.info(f'IMPORTANT: The mean of the standard scaler is {scaler_mean}')
logging.info(f'IMPORTANT: The standard deviation of the standard scaler is {scaler_std}')

#Splitting into input (i.e. the template parameters) and output (the match)
x_train = np.vstack((TrainingBank.ref_mass1.values, TrainingBank.ref_mass2.values, 
TrainingBank.mass1.values, TrainingBank.mass2.values,
TrainingBank.ref_spin1.values, TrainingBank.ref_spin2.values,
TrainingBank.spin1.values, TrainingBank.spin2.values)).T
y_train = TrainingBank.match.values

x_val = np.vstack((ValidationBank.ref_mass1.values, ValidationBank.ref_mass2.values, 
ValidationBank.mass1.values, ValidationBank.mass2.values,
ValidationBank.ref_spin1.values, ValidationBank.ref_spin2.values,
ValidationBank.spin1.values, ValidationBank.spin2.values)).T
y_val = ValidationBank.match.values

#Convert a numpy array to a Tensor
logging.info("Converting to datasets to a trainloader")
x_train = torch.tensor(x_train, dtype=torch.float32, device='cuda')
y_train = torch.tensor(y_train, dtype=torch.float32, device='cuda')
x_val = torch.tensor(x_val, dtype=torch.float32, device='cuda')
y_val = torch.tensor(y_val, dtype=torch.float32, device='cuda')

#Convert data into TensorDataset
xy_train = TensorDataset(x_train, y_train)
xy_val = TensorDataset(x_val, y_val)

#Convert data into DataLoaders
training_loader  = DataLoader(xy_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validation_loader = DataLoader(xy_val, batch_size=BATCH_SIZE, drop_last=True)

logging.info("Uploading the model")
our_model = NeuralNetwork().to(device)
criterion = torch.nn.MSELoss(reduction='sum') # return the sum so we can calculate the mse of the epoch.
optimizer = torch.optim.Adam(our_model.parameters(), lr = LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min')
logging.info("Model successfully loaded")

loss_list = []
val_list = []

epoch_number = 0
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
    print('EPOCH: {} TRAINING LOSS {} VALIDATION LOSS {}'.format(epoch_number, epoch_mse, vepoch_mse),end="\r")

    del epoch_loss
    del vepoch_loss
    del inputs
    del vinputs
    
    #This step to is determine whether the scheduler needs to be used 
    scheduler.step(vepoch_mse)

    #Create a loss_curve
    loss_list.append(epoch_mse)
    val_list.append(vepoch_mse)

    epoch_number += 1

    del epoch_mse
    del vepoch_mse

print("Time taken", time.time() - start_time)
writer.flush()
writer.close()

#A really complicated way of getting the list off the GPU and into a numpy array
loss_array = torch.tensor(loss_list, dtype=torch.float32, device='cpu').detach().numpy()
val_loss_array = torch.tensor(val_list, dtype=torch.float32, device='cpu').detach().numpy()

np.savetxt(TRAINING_LOSS, loss_array, delimiter=",")
np.savetxt(VALIDATION_LOSS, val_loss_array, delimiter=",")

#Save the Neural Network 
torch.save(our_model.state_dict(), LEARNINGMATCH_MODEL)


