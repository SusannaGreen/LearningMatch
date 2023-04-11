import numpy as np
import pandas as pd
import logging

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Define the location of the files that contain the training and validation loss for the three files
TRAINING_VALIDATION_LOSS_1 = r'/users/sgreen/LearningMatch/LVK/Paper/MassSpinParameters/ComparisonLossCurve/10000TrainingValidationLoss.csv'
TRAINING_VALIDATION_LOSS_2 = r'/users/sgreen/LearningMatch/LVK/Paper/MassSpinParameters/ComparisonLossCurve/100000TrainingValidationLoss.csv'
TRAINING_VALIDATION_LOSS_3 = r'/users/sgreen/LearningMatch/LVK/Paper/MassSpinParameters/ComparisonLossCurve/1000000TrainingValidationLoss.csv'

#Reading the test dataset
logging.info("Reading in the files")
training_validation_loss_1 = pd.read_csv(TRAINING_VALIDATION_LOSS_1)
training_validation_loss_2 = pd.read_csv(TRAINING_VALIDATION_LOSS_2)
training_validation_loss_3 = pd.read_csv(TRAINING_VALIDATION_LOSS_3)

training_loss_1 = training_validation_loss_1.training_loss.values
validation_loss_1 = training_validation_loss_1.vailidation_loss.values
training_loss_2 = training_validation_loss_2.training_loss.values
validation_loss_2 = training_validation_loss_2.vailidation_loss.values
training_loss_3 = training_validation_loss_3.training_loss.values
validation_loss_3 = training_validation_loss_3.vailidation_loss.values


#Plots the loss curve for training and validation data set
plt.figure(figsize=(8.2, 6.2))
plt.semilogy(np.arange(1, len(training_loss_1)+1), training_loss_1, color='#5B2C6F', label='10000 Training Loss')
plt.semilogy(np.arange(1, len(validation_loss_1)+1), validation_loss_1, color='#0096FF', label='10000 Validation Loss')
plt.semilogy(np.arange(1, len(training_loss_2)+1), training_loss_2, color='#5B2C6F', label='100000 Training Loss')
plt.semilogy(np.arange(1, len(validation_loss_2)+1), validation_loss_2, color='#0096FF', label='100000 Training Loss')
plt.semilogy(np.arange(1, len(training_loss_3)+1), training_loss_3, color='#5B2C6F', label='1000000 Training Loss')
plt.semilogy(np.arange(1, len(validation_loss_3)+1), validation_loss_3, color='#0096FF', label='1000000 Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('mass_spin_loss_curve.pdf')