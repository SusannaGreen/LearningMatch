#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew Lundgren, and Xan Morice-Atkinson 

import numpy as np
import pandas as pd

from LearningMatchModel.py import NeuralNetwork

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Define where to find the Test dataset
TEST_BANK = r'/users/sgreen/LearningMatch/Development/MassSpinTestBank.csv'
TRAINING_LOSS = r'/users/sgreen/LearningMatch/Development/TrainingLoss.csv'
VALIDATION_LOSS = r'/users/sgreen/LearningMatch/Development/ValidationLoss.csv'

#Define where to find the model
LEARNINGMATCH_MODEL = '/users/sgreen/LearningMatch/Development/LearningMatchModel.pth'

#Specify the Standard.Scaler() values which can be found on the log file from your LearningMatchTrain.py
SCALER_MEAN = 51 
SCALER_STD = 799

#Defining the outputs
LOSS_CURVE = '/users/sgreen/LearningMatch/Development/LossCurve.pdf'
ERROR_HISTOGRAM = '/users/sgreen/LearningMatch/Development/Error.pdf'
ACTUAL_PREDICTED_PLOT = '/users/sgreen/LearningMatch/Development/ActualPredicted.pdf'

#Defining the functions
def to_cpu_np(x):
    return x.cpu().detach().numpy()
    
def scaling_the_mass(mass):
    scaled_mass = (mass - SCALER_MEAN)/ np.sqrt(SCALER_STD)
    return scaled_mass

def rescaling_the_mass(mass):
    rescaled_mass = mass* np.sqrt(SCALER_STD) + SCALER_MEAN
    return rescaled_mass

#Reading the Training, validation and test dataset
logging.info("Reading in the data")
TestBank = pd.read_csv(TEST_BANK)
logging.info(f'The data size of the Test dataset is {len(TestBank)}, respectively')

#Convert to a Tensor
logging.info("Converting the test dataset into a tensor")
x_test = np.vstack((TestBank.ref_mass1.values, TestBank.ref_mass2.values, 
TestBank.mass1.values, TestBank.mass2.values,
TestBank.ref_spin1.values, TestBank.ref_spin2.values,
TestBank.spin1.values, TestBank.spin2.values)).T
y_test = TestBank.match.values

x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

#Upload the already trained weights and bias
logging.info("Loading the LearningMatch Model")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(LEARNINGMATCH_MODEL, map_location=device))
model.eval()

#Time taken to predict the match on the test dataset
logging.info("LearningMatch is predicting the Match for your dataset")  
with torch.no_grad():
    pred_start_time = time.time()
    y_prediction = model(x_test) 
    torch.cuda.synchronize()
    end_time = time.time()

print("Total time taken", end_time - pred_start_time)
print("Average time taken to predict the match", (end_time - pred_start_time)/len(x_test))

#A really complicated way of getting the list off the GPU and into a numpy array
loss_array = torch.tensor(loss_list, dtype=torch.float32, device='cpu').detach().numpy()
val_loss_array = torch.tensor(val_list, dtype=torch.float32, device='cpu').detach().numpy()

error = to_cpu_np(y_prediction[:, 0] - y_test)

#Plots the loss curve for training and validation data set
logging.info("Creating a Loss Curve")  
plt.figure(figsize=(8.2, 6.2))
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F', label='Training Loss')
plt.semilogy(np.arange(1, len(val_loss_array)+1), val_loss_array, color='#0096FF', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(LOSS_CURVE)

#Creates a Actual Match and Predicted Match plot with residuals
logging.info("Creating a Actual and Predicted Plot")  
fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 7), sharex=True, height_ratios=[3, 1])
ax1.scatter(x,y, s=8, color='#0096FF')
ax1.axline((0, 0), slope=1, color='k')
ax1.set_ylabel('Predicted Match')

sns.residplot(x=x, y=y, color = '#0096FF', scatter_kws={'s': 8}, line_kws={'linewidth':20})
ax2.set_ylabel('Error')

fig.supxlabel('Actual Match')
plt.savefig(ACTUAL_PREDICTED_PLOT, dpi=300)

#Creates plots to compare the errors
logging.info("Creating a Histogram of errors")  
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

