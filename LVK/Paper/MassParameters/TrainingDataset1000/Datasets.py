#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew Lundgren, and Xan Morice-Atkinson 
import numpy as np
import pandas as pd
import time
import logging 

from pycbc.filter import matchedfilter
from pycbc.psd.analytical import aLIGO140MpcT1800545
from pycbc.waveform import get_fd_waveform

#from joblib import Parallel, delayed
#results = Parallel(n_jobs=12, verbose=10)(delayed(process)(*arg) for arg in simlst)

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('Datasets.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define location of the output files 
DATA_DIR = '/users/sgreen/LearningMatch/LVK/Paper/MassParameters/TrainingDataset1000/'

#Define name of the output files
TRAINING_DATASET_FILE = DATA_DIR+'1000MassTrainingDataset.csv'
VALIDATION_DATASET_FILE = DATA_DIR+'100MassValidationDataset.csv'
TEST_DATASET_FILE = DATA_DIR+'5000MassTestDataset.csv'

#Define the size of the training, validation and test dataset
TRAINING_SIZE = 1000 #size of the training dataset
VALIDATION_SIZE = 100 #size of the validation dataset
TEST_SIZE = 5000 #size of the test dataset

#Define the detector
LOW_FREQ = 12 #frequency cut-off for the GW detector
SAMPLE_RATE = 4096 #sampling rate of desired detector
TLEN = 128
DELTA_F = 1.0 / TLEN
PSD = aLIGO140MpcT1800545(1+TLEN*SAMPLE_RATE//2, delta_f=DELTA_F, low_freq_cutoff=LOW_FREQ) 

#Define the template you want LearningMatch to learn
TEMPLATE = 'IMRPhenomXAS'

#Define the range of parameters you want LearningMatch to learn 
#If you require more parameters than the ones specified you will be required to adjust the script manually
MIN_MASS = 3.0
MAX_MASS = 100

#Define the functions
def dataset_generation(size_of_dataset, output_of_dataset):
    mass = np.random.uniform(MIN_MASS, MAX_MASS, size=(size_of_dataset, 2))
    reference_mass = np.random.uniform(MIN_MASS, MAX_MASS, size=(size_of_dataset, 2))

    parameters_list = []
    match_time = []
    start_time = time.time()
    for ref_m1, ref_m2, m1, m2 in zip(reference_mass[:, 0], reference_mass[:, 1], mass[:, 0], mass[:, 1]):
        template_generation = time.time()
        template_reference, _ = get_fd_waveform(approximant=TEMPLATE, mass1=ref_m1, mass2=ref_m2, delta_f=DELTA_F, f_lower=LOW_FREQ)
        template, _ = get_fd_waveform(approximant=TEMPLATE, mass1=m1, mass2=m2, delta_f=DELTA_F, f_lower=LOW_FREQ)
        template_reference.resize(len(PSD))
        template.resize(len(PSD))
        match, Index = template.match(template_reference, psd=PSD, low_frequency_cutoff=15)
        match_time.append(time.time()-template_generation)
        parameters_list.append([ref_m1, ref_m2, m1, m2, match])

    logger.info("Time taken to generate this dataset %s", time.time() - start_time)
    logger.info("Total time taken to calculate the match %s", sum(match_time))
    logger.info("The average time taken to calculate the match %s", sum(match_time)/len(match_time))

    MassSpinMatchDataset =  pd.DataFrame(data=(parameters_list), columns=['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'match'])
    MassSpinMatchDataset.to_csv(output_of_dataset, index = False)

#Generate the training, validation and test dataset
logger.info("Creating the training dataset")
dataset_generation(TRAINING_SIZE, TRAINING_DATASET_FILE)
logger.info("Creating the validation dataset")
dataset_generation(VALIDATION_SIZE, VALIDATION_DATASET_FILE)
logger.info("Creating the test dataset")
dataset_generation(TEST_SIZE, TEST_DATASET_FILE)