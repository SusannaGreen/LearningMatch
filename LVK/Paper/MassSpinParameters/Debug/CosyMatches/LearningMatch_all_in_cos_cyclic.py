#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

from Model import NeuralNetwork

import numpy as np
import pandas as pd
import time
import logging

from scipy.stats import gaussian_kde

from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CyclicLR

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('AllInOne.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define the functions
def to_cpu_np(x):
    return x.cpu().detach().numpy()

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/LearningMatch/LVK/Paper/MassSpinParameters/Debug/'

#Define location to the training and validation dataset
TRAINING_DATASET_FILE_PATH = DATA_DIR+r'1000000MassSpinCosyTrainingDataset.csv'
VALIDATION_DATASET_FILE_PATH = DATA_DIR+r'100000MassSpinCosyValidationDataset.csv'
TEST_DATASET_FILE_PATH = DATA_DIR+r'5000MassSpinCosyTestDataset.csv'

#Defining the location of the outputs
LOSS_CURVE = DATA_DIR+'1000000MassSpinLossCurveCosCyclic8.png'
ERROR_HISTOGRAM = DATA_DIR+'1000000MassSpinErrorCosCyclic8.png'
#ACTUAL_PREDICTED_PLOT = DATA_DIR+'1000000MassSpinActualPredictedLessEpochs2.png'

#Define output location of the LearningMatch model
LEARNINGMATCH_MODEL = DATA_DIR+'LearningMatchModelCosCyclic8.pth'

#Define values for the LearningMatch model
EPOCHS = 250
BATCH_SIZE = 64
LEARNING_RATE = 1e-5

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
                     TrainingBank.ref_spin1.values, TrainingBank.ref_spin2.values, 
                     TrainingBank.mass1.values, TrainingBank.mass2.values,
                     TrainingBank.spin1.values, TrainingBank.spin2.values)).T
y_train = TrainingBank.match.values

x_val = np.vstack((ValidationBank.ref_mass1.values, ValidationBank.ref_mass2.values,
                   ValidationBank.ref_spin1.values, ValidationBank.ref_spin2.values, 
                   ValidationBank.mass1.values, ValidationBank.mass2.values,
                   ValidationBank.spin1.values, ValidationBank.spin2.values)).T
y_val = ValidationBank.match.values

#Convert a numpy array to a Tensor
logger.info("Converting to datasets to a trainloader")
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
scheduler = CyclicLR(optimizer, base_lr = 1e-5, max_lr = 0.01, cycle_momentum=False )
compiled_model = torch.compile(our_model)
logger.info("Model successfully loaded")

start_time = time.time()

loss_list = []
val_list = []

epoch_number = 0
for epoch in range(EPOCHS):
    # Make sure gradient tracking is on, and do a pass over the data
    compiled_model.train(True)

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
    
    compiled_model.train(False)

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
    loss_list.append(epoch_mse)
    val_list.append(vepoch_mse)

    epoch_number += 1

    del epoch_mse
    del vepoch_mse

logger.info("Time taken to train LearningMatch %s", time.time() - start_time)

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")    

#Reading the test dataset
logger.info("Reading in the test dataset")
test_dataset = pd.read_csv(TEST_DATASET_FILE_PATH)
logger.info(f'The size of the test dataset is {len(test_dataset)}')

#Scaling the test dataset
logger.info("Scaling the test dataset")
test_dataset[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.transform(test_dataset[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])

#Convert to a Tensor
logger.info("Converting the test dataset into a tensor")
x_test = np.vstack((test_dataset.ref_mass1.values, test_dataset.ref_mass2.values,
                    test_dataset.ref_spin1.values, test_dataset.ref_spin2.values, 
                    test_dataset.mass1.values, test_dataset.mass2.values,
                    test_dataset.spin1.values, test_dataset.spin2.values)).T
y_test = test_dataset.match.values

x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

#Time taken to predict the match on the test dataset
logger.info("LearningMatch is predicting the Match for your dataset")  
with torch.no_grad():
    pred_start_time = time.time()
    y_prediction = compiled_model(x_test) 
    torch.cuda.synchronize()
    end_time = time.time()

logger.info(("Total time taken", end_time - pred_start_time))
logger.info(("Average time taken to predict the match", (end_time - pred_start_time)/len(x_test)))

#Plots the loss curve for training and validation data set
logger.info("Creating a loss curve which compares the training loss with validation loss")  

#A really complicated way of getting the list off the GPU and into a numpy array
loss_array = torch.tensor(loss_list, dtype=torch.float32, device='cpu').detach().numpy()
val_loss_array = torch.tensor(val_list, dtype=torch.float32, device='cpu').detach().numpy()

plt.figure(figsize=(9, 9))
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F', label='Training Loss')
plt.semilogy(np.arange(1, len(val_loss_array)+1), val_loss_array, color='#0096FF', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(LOSS_CURVE, dpi=300, bbox_inches='tight')

#Creates a plot that compares the actual match values with LearningMatch's predicted match values 
#logger.info("Creating a plot that compares tha actual match values with predicted match values, with residuals")  

#x = to_cpu_np(y_test)
#y = to_cpu_np(y_prediction[:, 0])

#fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 9), sharex=True, height_ratios=[3, 1])
#ax1.scatter(x,y, s=8, color='#0096FF')
#ax1.axline((0, 0), slope=1, color='k')
#ax1.set_ylabel('Predicted Match')

#sns.residplot(x=x, y=y, color = '#0096FF', scatter_kws={'s': 8}, line_kws={'linewidth':20})
#ax2.set_ylabel('Error')

#fig.supxlabel('Actual Match')
#plt.savefig(ACTUAL_PREDICTED_PLOT, dpi=300, bbox_inches='tight')

x = to_cpu_np(y_test)
y = to_cpu_np(y_prediction[:, 0])

#Creates a histogram of the errors
logger.info("Creating a histogram of the errors") 

error = to_cpu_np(y_prediction[:, 0] - y_test)

plt.figure(figsize=(9, 9))
plt.hist(error, bins=30, range=[error.min(), error.max()], color='#5B2C6F', align='mid', label='Errors for all match values')
plt.hist(error[x > .95], bins=30, range=[error.min(), error.max()], color='#0096FF', align='mid', label='Errors for match values over 0.95')
plt.xlim([error.min(), error.max()])
plt.xticks([-0.05, -0.01, 0.01, 0.05])
plt.yscale('log')
plt.xlabel('Error')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.savefig(ERROR_HISTOGRAM, dpi=300, bbox_inches='tight')

#Save the trained LearningMatch model 
torch.save(our_model.state_dict(), LEARNINGMATCH_MODEL)

#Removing predictions great than 1 so it can be converted into cosine
converted_x = []
converted_y = []
actual_error = []
for i, j in zip(x, y):
    if 0<=j<=1:
        actual_y_test = np.arccos(1-i)*(2/np.pi)
        actual_y_predictedion = np.arccos(1-j)*(2/np.pi)
        converted_x.append(actual_y_test) #np.arccos(y_test)*(4/np.pi)
        converted_y.append(actual_y_predictedion)
        actual_error.append(actual_y_predictedion - actual_y_test)
    else: 
        pass

#Converting to an array
converted_x = np.array([converted_x])[0, :]
converted_y = np.array([converted_y])[0, :]
actual_error = np.array([actual_error])[0, :]

#Creates plots to compare the actual and predicted matches 
plt.figure()

# Calculate the point density
xy = np.vstack([x,y]) 
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx1 = z.argsort()
x, y, z = x[idx1], y[idx1], z[idx1]

plt.scatter(x,y,c=z, s=20)
plt.xlabel('Actual cosy match')
plt.ylabel('Predicted cosy match')
plt.savefig('cyclic_cosy_actual_predicted8.png', dpi=300)

#Plots the distribution of errors for the test data
plt.figure()
plt.hist(error, bins=50, color='#5B2C6F')
plt.xlabel('Error')
plt.ylabel('Count')
plt.savefig('cyclic_cosy_error8.png', dpi=300)

#Plots the distribution of errors for the test data where the actual match > .95
plt.figure()
plt.hist(error[x > .95], bins=50, color='#5B2C6F')
plt.xlabel('Error')
plt.ylabel('Count')
plt.savefig('cyclic_cosy_error_clipped8.png', dpi=300)

#Creates plots to compare the actual and predicted matches 
plt.figure()

# Calculate the point density
ab = np.vstack([converted_x, converted_y])
c = gaussian_kde(ab)(ab)

# Sort the points by density, so that the densest points are plotted last
idx2 = c.argsort()
a, b, c = converted_x[idx2], converted_y[idx2], c[idx2]

plt.scatter(converted_x, converted_y, c=c, s=20)
plt.xlabel('Actual match')
plt.ylabel('Predicted match')
plt.savefig('cyclic_actual_predicted8.png', dpi=300)

plt.figure()
plt.scatter(converted_x, converted_y, color='#5B2C6F')
plt.xlabel('Actual match')
plt.ylabel('Predicted match')
plt.savefig('cyclic_actual_predicted_presentation8.png', dpi=300)

#Plots the distribution of errors for the test data
plt.figure()
plt.hist(actual_error, bins=50, color='#5B2C6F')
plt.xlabel('Error')
plt.ylabel('Count')
plt.savefig('cyclic_error8.png', dpi=300)

#Plots the distribution of errors for the test data where the actual match > .95
plt.figure()
plt.hist(actual_error[converted_x > .95], bins=50, color='#5B2C6F')
plt.xlabel('Error')
plt.ylabel('Count')
plt.savefig('cyclic_error_clipped8.png', dpi=300)


