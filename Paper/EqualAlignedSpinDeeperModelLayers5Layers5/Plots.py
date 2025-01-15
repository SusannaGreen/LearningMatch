#!/usr/bin/env python

# Copyright (C) 2024 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

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

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('Plots.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/LearningMatch/Paper/EqualAlignedSpinDeeperModelLayers5Layers5/'

#Define location of the test dataset
TEST_DATASET_FILE_PATH = '/users/sgreen/LearningMatch/Paper/EqualAlignedSpinDeeperModelLayers5Layers5/100000LambdaEtaAlignedSpinTestDataset+150000DiffusedLambdaEtaAlignedSpinTestDataset.csv'

#Define location of the trained LearningMatch model 
LEARNINGMATCH_MODEL =  DATA_DIR+'LearningMatchModel.pth'

#Define location of the loss File
LOSS_FILE = DATA_DIR+ r'TrainingValidationLoss.csv'

#Defining the location of the outputs
LOSS_CURVE = DATA_DIR+'LossCurve.png'
ERROR_HISTOGRAM = DATA_DIR+'Error.png'
ACTUAL_PREDICTED_PLOT = DATA_DIR+'ActualPredicted.png'
        
#Define the functions
def to_cpu_np(x):
    return x.cpu().detach().numpy()

def my_collate_fn(data):
    return tuple(data)

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")    

#Reading the test dataset
logger.info("Reading in the test dataset")
test_dataset = pd.read_csv(TEST_DATASET_FILE_PATH)
logger.info(f'The size of the test dataset is {len(test_dataset)}')

#Convert to a Tensor
logger.info("Converting the test dataset into a tensor")
x_test = np.vstack((test_dataset.ref_lambda0.values, test_dataset.ref_eta.values,
                    test_dataset.ref_spin1.values, test_dataset.ref_spin2.values, 
                    test_dataset.lambda0.values, test_dataset.eta.values,
                    test_dataset.spin1.values, test_dataset.spin2.values)).T
y_test = test_dataset.match.values.reshape(-1,1)

x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

#Convert data into TensorDataset
xy_test = MyDataset(x_test, y_test)

#Convert data into DataLoaders
test_data_loader  = DataLoader(xy_test, collate_fn=my_collate_fn, batch_size=1024, shuffle=False)

#Upload the already trained weights and bias
logger.info("Loading the LearningMatch model")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(LEARNINGMATCH_MODEL, map_location=device))
model.eval()

compiled_model = torch.compile(model)

x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

error_test_dataset = torch.zeros((1,1), dtype=torch.float64, device=device)
y_test_dataset = torch.zeros((1,1), dtype=torch.float64, device=device)
y_prediction_dataset = torch.zeros((1,1), dtype=torch.float64, device=device)

#Time taken to predict the match on the test dataset
logger.info("LearningMatch is predicting the Match for your dataset")
with torch.inference_mode():
    pred_start_time = time.time()
    for i, data in enumerate(test_data_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Make predictions for this batch
        y_prediction = compiled_model(inputs)
        error = labels - y_prediction
        error_test_dataset = torch.cat((error_test_dataset, error.detach()), 0)
        y_test_dataset = torch.cat((y_test_dataset, labels.detach()), 0)
        y_prediction_dataset = torch.cat((y_prediction_dataset, y_prediction.detach()), 0)
    torch.cuda.synchronize()
    end_time = time.time()
  
logger.info(("Total time taken", end_time - pred_start_time))
logger.info(("Average time taken to predict the match", (end_time - pred_start_time)/len(x_test)))

#Plots the loss curve for training and validation data set
logger.info("Creating a loss curve which compares the training loss with validation loss")  

Loss = pd.read_csv(LOSS_FILE)
validation_loss = Loss.validation_loss.values
training_loss = Loss.training_loss.values

plt.figure(figsize=(9, 9))
plt.semilogy(np.arange(1, len(training_loss)+1), training_loss, color='#5B2C6F', label='Training Loss')
plt.semilogy(np.arange(1, len(validation_loss)+1), validation_loss, color='#0096FF', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(LOSS_CURVE, dpi=300, bbox_inches='tight')

#Creates a plot that compares the actual match values with LearningMatch's predicted match values 
logger.info("Creating a plot that compares tha actual match values with predicted match values, with residuals")  

x = to_cpu_np(y_test_dataset)
y = to_cpu_np(y_prediction_dataset)

fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 9), sharex=True, height_ratios=[3, 1])
ax1.scatter(x,y, s=8, color='#4690ef')
ax1.axline((0, 0), slope=1, color='k')
ax1.set_ylabel('Predicted Match')

sns.residplot(x=x, y=y, color = '#4690ef', scatter_kws={'s': 8}, line_kws={'linewidth':20})
ax2.set_ylabel('Error')

fig.supxlabel('Actual Match')
plt.savefig(ACTUAL_PREDICTED_PLOT, dpi=300, bbox_inches='tight')

#Creates a histogram of the errors
logger.info("Creating a histogram of the errors") 

error = to_cpu_np(error_test_dataset)

logger.info(("The maximum error", np.max(error)))
logger.info(("The maximum error for values greater than 0.95", np.max(error[x > .95])))

plt.figure(figsize=(9, 9))
plt.hist(error, bins=30, range=[error.min(), error.max()], color='#a74b9cff', align='mid', label='Errors for all match values')
plt.hist(error[x > .95], bins=30, range=[error.min(), error.max()], color='#4690ef', align='mid', label='Errors for match values over 0.95')
plt.xlim([error.min(), error.max()])
plt.xticks([-0.1, -0.05, -0.01, 0.01, 0.05, 0.1])
plt.yscale('log')
plt.xlabel('$\delta\mathcal{M}$')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.savefig(ERROR_HISTOGRAM, dpi=300, bbox_inches='tight')
