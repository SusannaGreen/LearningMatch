#!/usr/bin/env python

# Copyright (C) 2025 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

from Model import NeuralNetwork
from Dataset import FastBatchSampler, FastRandomSampler, MyDataset

import numpy as np
import pandas as pd
import time
import logging
from typing import Iterator, List, Sized

import torch
import torch.nn as nn
from torch.utils.data.dataset import Tensor
from torch.utils.data.sampler import Sampler 
from torch.utils.data import DataLoader, TensorDataset

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('Train.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/LearningMatch/Paper/EqualAlignedSpinTrainingDataset250000/'

#Define location to the training and validation dataset
TRAINING_DATASET_FILE_PATH = DATA_DIR+r'100000LambdaEtaAlignedSpinTrainingDataset+150000DiffusedLambdaEtaAlignedSpinTrainingDataset.csv'
VALIDATION_DATASET_FILE_PATH = DATA_DIR+r'10000LambdaEtaAlignedSpinValidationDataset+15000DiffusedLambdaEtaAlignedSpinValidationDataset.csv'

#Define output location of the LearningMatch model
LEARNINGMATCH_MODEL = DATA_DIR+'LearningMatchModel.pth'

#Define output location for the training and validation loss
LOSS = DATA_DIR+'TrainingValidationLoss.csv'

#Define values for the LearningMatch model
EPOCHS = 5000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-6

#Define Functions
def my_collate_fn(data):
    return tuple(data)

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")

#Reading the Training, validation and test dataset
logger.info("Reading in the data")
TrainingBank = pd.read_csv(TRAINING_DATASET_FILE_PATH)
ValidationBank = pd.read_csv(VALIDATION_DATASET_FILE_PATH)
logger.info(f'The data size of the training and validation dataset is {len(TrainingBank), len(ValidationBank)}, respectively')

#Splitting into input (i.e. the template parameters) and output (the match)
x_train = np.vstack((TrainingBank.ref_lambda0.values, TrainingBank.ref_eta.values, 
                     TrainingBank.ref_spin1.values, TrainingBank.ref_spin2.values, 
                     TrainingBank.lambda0.values, TrainingBank.eta.values,
                     TrainingBank.spin1.values, TrainingBank.spin2.values)).T
y_train = TrainingBank.match.values.reshape(-1,1)

x_val = np.vstack((ValidationBank.ref_lambda0.values, ValidationBank.ref_eta.values,
                   ValidationBank.ref_spin1.values, ValidationBank.ref_spin2.values, 
                   ValidationBank.lambda0.values, ValidationBank.eta.values,
                   ValidationBank.spin1.values, ValidationBank.spin2.values)).T
y_val = ValidationBank.match.values.reshape(-1,1)

#Convert a numpy array to a Tensor
logger.info("Converting to datasets to a trainloader")
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)#.reshape(1, *y_train.shape).T
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)#.reshape(1, *y_val.shape).T

#Convert data into TensorDataset
xy_train = MyDataset(x_train, y_train)
xy_val = MyDataset(x_val, y_val)

del x_train
del y_train
del x_val
del y_val

sampler = FastBatchSampler(FastRandomSampler(xy_train), BATCH_SIZE, drop_last=True)
vsampler = FastBatchSampler(FastRandomSampler(xy_val), BATCH_SIZE, drop_last=True)

#Convert data into DataLoaders
training_loader  = DataLoader(xy_train, collate_fn=my_collate_fn, sampler=sampler, num_workers=0, pin_memory=True)
validation_loader = DataLoader(xy_val, collate_fn=my_collate_fn, sampler=vsampler, num_workers=0, pin_memory=True)

logger.info("Uploading the model")
our_model = NeuralNetwork().to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(our_model.parameters(), lr = LEARNING_RATE)
compiled_model = torch.compile(our_model)
logger.info("Model successfully loaded")

for parameter in our_model.parameters():
    logger.info(len(parameter))

trainable_params =sum(p.numel()for p in our_model.parameters()if p.requires_grad)
logger.info(f'The number of trainable parameters is {trainable_params}')

start_time = time.time()

loss_list = torch.zeros(1, dtype=torch.float64, device=device)
val_list = torch.zeros(1, dtype=torch.float64, device=device)

nbatches = len(training_loader)
val_nbatches = len(validation_loader)

epoch_number = 0
for epoch in range(EPOCHS):
    # Make sure gradient tracking is on, and do a pass over the data
    compiled_model.train(True)

    iters  = torch.zeros(1, dtype=torch.float64, device=device)
    epoch_loss = torch.zeros(1, dtype=torch.float64, device=device)
    val_iters = torch.zeros(1, dtype=torch.float64, device=device)
    vepoch_loss = torch.zeros(1, dtype=torch.float64, device=device)

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        outputs = compiled_model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        del outputs
        del labels

        # save the current training information
        epoch_loss += loss.detach()
            
    compiled_model.train(False)

    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.to(device, non_blocking=True)
        vlabels = vlabels.to(device, non_blocking=True)
        voutputs = compiled_model(vinputs)
        vloss = criterion(voutputs, vlabels)

        del voutputs
        del vlabels

        vepoch_loss += vloss.detach()

    epoch_mse = epoch_loss/nbatches
    vepoch_mse = vepoch_loss/val_nbatches
    logger.info('EPOCH: {} TRAINING LOSS {} VALIDATION LOSS {}'.format(epoch_number, float(epoch_mse.detach()), float(vepoch_mse.detach())))
    
    #scheduler.step(vepoch_mse)
    
    del epoch_loss
    del vepoch_loss
    del inputs
    del vinputs

    loss_list = torch.cat((loss_list, epoch_mse.detach()), 0)
    val_list = torch.cat((val_list, vepoch_mse.detach()), 0)

    del epoch_mse
    del vepoch_mse

    epoch_number += 1

    #Checkpoint every 10 epochs
    if epoch_number % 10 == 0:
        torch.save(our_model.state_dict(), LEARNINGMATCH_MODEL)
        logger.info("Model saved at epoch %s", epoch_number)

logger.info("Time taken to train LearningMatch %s", time.time() - start_time)

#Creates a file with the training and validation loss
loss_array = torch.tensor(loss_list, dtype=torch.float32, device='cpu').detach().numpy() #Gets the list off the GPU and into a numpy array
val_loss_array = torch.tensor(val_list, dtype=torch.float32, device='cpu').detach().numpy() #Gets the list off the GPU and into a numpy array
loss_array = np.delete(loss_array, 0)
val_loss_array = np.delete(val_loss_array, 0)
MassSpinMatchDataset =  pd.DataFrame(data=(np.vstack((loss_array, val_loss_array)).T), columns=['training_loss', 'validation_loss'])
MassSpinMatchDataset.to_csv(LOSS, index = False)

#Save the trained LearningMatch model 
torch.save(our_model.state_dict(), LEARNINGMATCH_MODEL)


