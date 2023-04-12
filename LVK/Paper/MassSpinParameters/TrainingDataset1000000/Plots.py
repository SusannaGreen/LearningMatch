#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

from Model import NeuralNetwork

import numpy as np
import pandas as pd
import seaborn as sns
import logging
import time 

from sklearn import preprocessing
from joblib import load

import torch
import torch.nn as nn

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
DATA_DIR = '/users/sgreen/LearningMatch/LVK/Paper/MassSpinParameters/TrainingDataset1000000/'

#Define location of the test dataset
TEST_DATASET_FILE_PATH = DATA_DIR+r'5000MassTestDataset.csv'

#Define location of the scaling
SCALER_FILE_PATH = DATA_DIR+'StandardScaler.bin'

#Define location of the trained LearningMatch model 
LEARNINGMATCH_MODEL =  DATA_DIR+'LearningMatchModel.pth'

#Define location of the loss File
LOSS_FILE = DATA_DIR+ r'TrainingValidationLoss.csv'

#Define the functions
def to_cpu_np(x):
    return x.cpu().detach().numpy()

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")    

#Reading the test dataset
logger.info("Reading in the test dataset")
test_dataset = pd.read_csv(TEST_DATASET_FILE_PATH)
logger.info(f'The size of the test dataset is {len(test_dataset)}')

#Scaling the test dataset
logger.info("Scaling the test dataset")
scaler = load(SCALER_FILE_PATH)
test_dataset[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.transform(test_dataset[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])

#Convert to a Tensor
logger.info("Converting the test dataset into a tensor")
x_test = np.vstack((test_dataset.ref_mass1.values, test_dataset.ref_mass2.values, 
test_dataset.mass1.values, test_dataset.mass2.values,
test_dataset.ref_spin1.values, test_dataset.ref_spin2.values,
test_dataset.spin1.values, test_dataset.spin2.values)).T
y_test = test_dataset.match.values

x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

#Upload the already trained weights and bias
logger.info("Loading the LearningMatch model")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(LEARNINGMATCH_MODEL, map_location=device))
model.eval()

#Time taken to predict the match on the test dataset
logger.info("LearningMatch is predicting the Match for your dataset")  
with torch.no_grad():
    pred_start_time = time.time()
    y_prediction = model(x_test) 
    torch.cuda.synchronize()
    end_time = time.time()

logger.info(("Total time taken", end_time - pred_start_time))
logger.info(("Average time taken to predict the match", (end_time - pred_start_time)/len(x_test)))

#Plots the loss curve for training and validation data set
logger.info("Creating a loss curve which compares the training loss with validation loss")  

Loss = pd.read_csv(LOSS_FILE)
validation_loss = Loss.validation_loss.values
training_loss = Loss.training_loss.values

plt.figure(figsize=(8.2, 6.2))
plt.semilogy(np.arange(1, len(training_loss)+1), training_loss, color='#5B2C6F', label='Training Loss')
plt.semilogy(np.arange(1, len(validation_loss)+1), validation_loss, color='#0096FF', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(LOSS_CURVE)

#Creates a plot that compares the actual match values with LearningMatch's predicted match values 
logger.info("Creating a plot that compares tha actual match values with predicted match values, with residuals")  

x = to_cpu_np(y_test)
y = to_cpu_np(y_prediction[:, 0])

fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 7), sharex=True, height_ratios=[3, 1])
ax1.scatter(x,y, s=8, color='#0096FF')
ax1.axline((0, 0), slope=1, color='k')
ax1.set_ylabel('Predicted Match')

sns.residplot(x=x, y=y, color = '#0096FF', scatter_kws={'s': 8}, line_kws={'linewidth':20})
ax2.set_ylabel('Error')

fig.supxlabel('Actual Match')
plt.savefig(ACTUAL_PREDICTED_PLOT, dpi=300)

#Creates a histogram of the errors
logger.info("Creating a histogram of the errors") 

error = to_cpu_np(y_prediction[:, 0] - y_test)

plt.figure(figsize=(9, 7))
plt.hist(error, bins=30, range=[error.min(), error.max()], color='#5B2C6F', align='mid', label='Errors for all match values')
plt.hist(error[x > .95], bins=30, range=[error.min(), error.max()], color='#0096FF', align='mid', label='Errors for match values over 0.95')
plt.xlim([error.min(), error.max()])
plt.xticks([-0.1, -0.05, -0.01, 0.01, 0.05, 0.1])
plt.yscale('log')
plt.xlabel('Error')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.savefig(ERROR_HISTOGRAM, dpi=300)

