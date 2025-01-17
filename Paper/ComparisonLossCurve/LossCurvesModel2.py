#!/usr/bin/env python

# Copyright (C) 2025 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 
import numpy as np
import pandas as pd
import logging

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('LossCurveModel2.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define the location of the files that contain the training and validation loss for the three files
TRAINING_VALIDATION_LOSS_2 = r'/users/sgreen/LearningMatch/Paper/DeeperModelLayers4Layers2/TrainingValidationLoss.csv'
TRAINING_VALIDATION_LOSS_3 = r'/users/sgreen/LearningMatch/Paper/DeeperModelLayers4Layers3/TrainingValidationLoss.csv'
TRAINING_VALIDATION_LOSS_4 = r'/users/sgreen/LearningMatch/Paper/DeeperModelLayers4Layers4/TrainingValidationLoss.csv'
TRAINING_VALIDATION_LOSS_5 = r'/users/sgreen/LearningMatch/Paper/DeeperModelLayers4Layers5/TrainingValidationLoss.csv'
TRAINING_VALIDATION_LOSS_6 = r'/users/sgreen/LearningMatch/Paper/DeeperModelLayers4Layers6/TrainingValidationLoss.csv'

#Reading the test dataset
logging.info("Reading in the files")
training_validation_loss_2 = pd.read_csv(TRAINING_VALIDATION_LOSS_2)
training_validation_loss_3 = pd.read_csv(TRAINING_VALIDATION_LOSS_3)
training_validation_loss_4 = pd.read_csv(TRAINING_VALIDATION_LOSS_4)
training_validation_loss_5 = pd.read_csv(TRAINING_VALIDATION_LOSS_5)
training_validation_loss_6 = pd.read_csv(TRAINING_VALIDATION_LOSS_6)

training_loss_2 = training_validation_loss_2.training_loss.values[25:]
validation_loss_2 = training_validation_loss_2.validation_loss.values
training_loss_3 = training_validation_loss_3.training_loss.values[25:]
validation_loss_3 = training_validation_loss_3.validation_loss.values
training_loss_4 = training_validation_loss_4.training_loss.values[25:]
validation_loss_4 = training_validation_loss_4.validation_loss.values
training_loss_5 = training_validation_loss_5.training_loss.values[25:]
validation_loss_5 = training_validation_loss_5.validation_loss.values
training_loss_6 = training_validation_loss_6.training_loss.values[25:]
validation_loss_6 = training_validation_loss_6.validation_loss.values[25:]


#Plots the loss curve for training and validation data set
plt.figure(figsize=(9, 9))
plt.semilogy(np.arange(1, len(training_loss_2)+1), training_loss_2, color='#8855fc', label='2 crunch layers')
plt.semilogy(np.arange(1, len(training_loss_3)+1), training_loss_3, color='#1c9df8', label='3 crunch layers')
plt.semilogy(np.arange(1, len(training_loss_4)+1), training_loss_4, color='#5B2C6F', label='4 crunch layers')
plt.semilogy(np.arange(1, len(training_loss_5)+1), training_loss_5, color='#b00068', label='5 crunch layers')
plt.semilogy(np.arange(1, len(training_loss_6)+1), training_loss_6, color='#fd3db5', label='6 crunch layers')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(fontsize="small")
plt.savefig('LossCurveModel2.png', dpi=300, bbox_inches='tight')