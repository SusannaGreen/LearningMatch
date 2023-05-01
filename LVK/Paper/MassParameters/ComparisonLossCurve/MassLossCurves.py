import numpy as np
import pandas as pd
import logging

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Gets or creates a logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('LossCurve.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

#Define the location of the files that contain the training and validation loss for the three files
TRAINING_VALIDATION_LOSS_1 = r'/users/sgreen/LearningMatch/LVK/Paper/MassParameters/ComparisonLossCurve/1000TrainingValidationLoss.csv'
TRAINING_VALIDATION_LOSS_2 = r'/users/sgreen/LearningMatch/LVK/Paper/MassParameters/ComparisonLossCurve/10000TrainingValidationLoss.csv'
TRAINING_VALIDATION_LOSS_3 = r'/users/sgreen/LearningMatch/LVK/Paper/MassParameters/ComparisonLossCurve/100000TrainingValidationLoss.csv'

#Reading the test dataset
logger.info("Reading in the files")
training_validation_loss_1 = pd.read_csv(TRAINING_VALIDATION_LOSS_1)
training_validation_loss_2 = pd.read_csv(TRAINING_VALIDATION_LOSS_2)
training_validation_loss_3 = pd.read_csv(TRAINING_VALIDATION_LOSS_3)

training_loss_1 = training_validation_loss_1.training_loss.values
validation_loss_1 = training_validation_loss_1.validation_loss.values
training_loss_2 = training_validation_loss_2.training_loss.values
validation_loss_2 = training_validation_loss_2.validation_loss.values
training_loss_3 = training_validation_loss_3.training_loss.values
validation_loss_3 = training_validation_loss_3.validation_loss.values


#Plots the loss curve for training and validation data set
plt.figure(figsize=(9, 9))
plt.semilogy(np.arange(1, len(training_loss_1)+1), training_loss_1, color='#5B2C6F', label='10000 Training Loss')
plt.semilogy(np.arange(1, len(validation_loss_1)+1), validation_loss_1, color='#0096FF', label='10000 Validation Loss')
plt.semilogy(np.arange(1, len(training_loss_2)+1), training_loss_2, color='#5B2C6F', label='100000 Training Loss')
plt.semilogy(np.arange(1, len(validation_loss_2)+1), validation_loss_2, color='#0096FF', label='100000 Validation Loss')
plt.semilogy(np.arange(1, len(training_loss_3)+1), training_loss_3, color='#5B2C6F', label='1000000 Training Loss')
plt.semilogy(np.arange(1, len(validation_loss_3)+1), validation_loss_3, color='#0096FF', label='1000000 Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(fontsize="small")
plt.savefig('MassLossCurve.png', dpi=300, bbox_inches='tight')