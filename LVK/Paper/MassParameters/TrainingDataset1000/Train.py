#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

from Model import NeuralNetwork

import numpy as np
import pandas as pd
import time
import logging

from sklearn import preprocessing
from joblib import dump

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('Train.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/LearningMatch/LVK/Paper/MassParameters/TrainingDataset1000/'

#Define location to the training and validation dataset
TRAINING_DATASET_FILE_PATH = DATA_DIR+r'1000MassTrainingDataset.csv'
VALIDATION_DATASET_FILE_PATH = DATA_DIR+r'100MassValidationDataset.csv'

#Define ouput location of the Standard.Scaler()
STANDARD_SCALER = DATA_DIR+'StandardScaler.bin'

#Define output location of the LearningMatch model
LEARNINGMATCH_MODEL = DATA_DIR+'LearningMatchModel.pth'

#Define output location for the training and validation loss
LOSS = DATA_DIR+'1000TrainingValidationLoss.csv'

#Define values for the LearningMatch model
EPOCHS = 400
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")

#Reading the Training, validation and test dataset
logger.info("Reading in the data")
TrainingBank = pd.read_csv(TRAINING_DATASET_FILE_PATH)
ValidationBank = pd.read_csv(VALIDATION_DATASET_FILE_PATH)
logger.info(f'The data size of the training and validation dataset is {len(TrainingBank), len(ValidationBank)}, respectively')

#Using Standard.scalar() to re-scale the Training Bank and applying to the validation and test bank. 
scaler = preprocessing.StandardScaler()
TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.fit_transform(TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])
ValidationBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.transform(ValidationBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])

#Splitting into input (i.e. the template parameters) and output (the match)
x_train = np.vstack((TrainingBank.ref_mass1.values, TrainingBank.ref_mass2.values, 
TrainingBank.mass1.values, TrainingBank.mass2.values)).T
y_train = TrainingBank.match.values

x_val = np.vstack((ValidationBank.ref_mass1.values, ValidationBank.ref_mass2.values, 
ValidationBank.mass1.values, ValidationBank.mass2.values)).T
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

logger.info("Uploading the model")
our_model = NeuralNetwork().to(device)
criterion = torch.nn.MSELoss(reduction='sum') # return the sum so we can calculate the mse of the epoch.
optimizer = torch.optim.Adam(our_model.parameters(), lr = LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min')
compiled_model = torch.compile(our_model)
logger.info("Model successfully loaded")

loss_list = []

epoch_number = 0
for epoch in range(EPOCHS):
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
        outputs = compiled_model(inputs)

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
        voutputs = compiled_model(vinputs)
        vloss = criterion(voutputs, vlabels)

        del voutputs
        del vlabels

        val_iters.append(i)
        vbatch_mse = vloss/vinputs.size(0)
        del vbatch_mse
        vepoch_loss += float(vloss)

    epoch_mse = epoch_loss/(len(iters) * inputs.size(0))
    vepoch_mse = vepoch_loss/(len(val_iters) * vinputs.size(0))
    logger.info('EPOCH: {} TRAINING LOSS {} VALIDATION LOSS {}'.format(epoch_number, epoch_mse, vepoch_mse))

    del epoch_loss
    del vepoch_loss
    del inputs
    del vinputs
    
    #This step to is determine whether the scheduler needs to be used 
    scheduler.step(vepoch_mse)

    #Create a loss_curve
    loss_list.append([epoch_mse, vepoch_mse])

    epoch_number += 1

    del epoch_mse
    del vepoch_mse

#Creates a file with the training and validation loss
loss_array = torch.tensor(loss_list, dtype=torch.float32, device='cpu').detach().numpy() #Gets the list off the GPU and into a numpy array
MassSpinMatchDataset =  pd.DataFrame(data=(loss_array), columns=['training_loss', 'validation_loss'])
MassSpinMatchDataset.to_csv(LOSS, index = False)

#Save the trained LearningMatch model 
torch.save(our_model.state_dict(), LEARNINGMATCH_MODEL)

#Save the StandardScaler.()
dump(scaler, STANDARD_SCALER, compress=True)


