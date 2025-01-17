#!/usr/bin/env python

# Copyright (C) 2025 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

import numpy as np
from numpy.random import default_rng
import pandas as pd
import time
import logging

from pycbc.filter import match as pycbc_match
from pycbc.psd.analytical import aLIGO140MpcT1800545
from pycbc.waveform import get_fd_waveform
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta

from joblib import Parallel, delayed
import joblib

#Define directory for the output
DATA_DIR = '/users/sgreen/LearningMatch/Paper/DatasetGeneration/'

#Define the dataset training, validation and test dataset
DATASET_FILE = DATA_DIR+'15000DiffusedLambdaEtaAlignedSpinTrainingDataset.csv'

MIN_MCHIRP = 5.0
MAX_MCHIRP = 20.0
REF_LAMBDA = 5.0
MIN_LAMBDA = (MAX_MCHIRP/REF_LAMBDA)**(-5/3)
MAX_LAMBDA = (MIN_MCHIRP/REF_LAMBDA)**(-5/3)
MIN_ETA = 0.1
MAX_ETA = 0.249999
MIN_SPIN = -0.99
MAX_SPIN = 0.99

SD_LAMBDA1 = 0.0001
SD_ETA1 = 0.01
SD_SPIN1 = 0.01

SD_LAMBDA2 = 0.001
SD_ETA2 = 0.01
SD_SPIN2 = 0.01

SD_LAMBDA3 = 0.01
SD_ETA3 = 0.02
SD_SPIN3 = 0.01

#Define the detector
LOW_FREQ = 15 #frequency cut-off for the GW detector
MATCH_LOW_FREQ = 18
SAMPLE_RATE = 2048 #sampling rate of desired detector
TLEN = 32
DELTA_F = 1.0 / TLEN
PSD = aLIGO140MpcT1800545(1+TLEN*SAMPLE_RATE//2, delta_f=DELTA_F, low_freq_cutoff=LOW_FREQ) 

#Define the template you want LearningMatch to learn
TEMPLATE = 'IMRPhenomXAS'

SIZE_OF_DATASET = 5

#Define the functions
def lambda0_to_mchirp(lambda0):
    return (lambda0**(-3/5))*REF_LAMBDA

def mchirp_to_lambda0(mchirp): 
    return (mchirp/REF_LAMBDA)**(-5/3)

def draw_lambda(rng, n_sims):
    return rng.uniform(MIN_LAMBDA,
                        MAX_LAMBDA,
                        size=n_sims)

def draw_eta(rng, n_sims):
    return rng.uniform(MIN_ETA, MAX_ETA,
                        size=n_sims)

def draw_spin(rng, n_sims):
    return rng.uniform(MIN_SPIN, MAX_SPIN,
                        size=n_sims)

def create_cbc_list(rng, n_sims):
    ref_lambda0 = draw_lambda(rng, n_sims)
    lambda01 = ref_lambda0 + rng.normal(0., SD_LAMBDA1,
                                        size=n_sims)
    lambda02 = ref_lambda0 + rng.normal(0., SD_LAMBDA2,
                                        size=n_sims)
    lambda03 = ref_lambda0 + rng.normal(0., SD_LAMBDA3,
                                        size=n_sims)
    
    lambda01[lambda01 < MIN_LAMBDA] = MIN_LAMBDA
    lambda01[lambda01 > MAX_LAMBDA] = MAX_LAMBDA
    lambda02[lambda02 < MIN_LAMBDA] = MIN_LAMBDA
    lambda02[lambda02 > MAX_LAMBDA] = MAX_LAMBDA
    lambda03[lambda03 < MIN_LAMBDA] = MIN_LAMBDA
    lambda03[lambda03 > MAX_LAMBDA] = MAX_LAMBDA

    ref_eta = draw_eta(rng, n_sims)
    eta1 = ref_eta + rng.normal(0., SD_ETA1,
                                        size=n_sims)
    eta2 = ref_eta + rng.normal(0., SD_ETA2,
                                        size=n_sims)
    eta3 = ref_eta + rng.normal(0., SD_ETA3,
                                        size=n_sims)
    eta1[eta1 < MIN_ETA] = MIN_ETA
    eta1[eta1 > MAX_ETA] = MAX_ETA
    eta2[eta2 < MIN_ETA] = MIN_ETA
    eta2[eta2 > MAX_ETA] = MAX_ETA
    eta3[eta3 < MIN_ETA] = MIN_ETA
    eta3[eta3 > MAX_ETA] = MAX_ETA

    ref_spin = draw_spin(rng, n_sims)

    spin1 = ref_spin + rng.normal(0., SD_SPIN1,
                                        size=n_sims)
    spin2 = ref_spin + rng.normal(0., SD_SPIN2,
                                        size=n_sims)
    spin3 = ref_spin + rng.normal(0., SD_SPIN3,
                                        size=n_sims)
    spin1[spin1 < MIN_SPIN] = MIN_SPIN
    spin2[spin2 < MIN_SPIN] = MIN_SPIN
    spin3[spin3 < MIN_SPIN] = MIN_SPIN
    spin1[spin1 > MAX_SPIN] = MIN_SPIN
    spin2[spin2 > MAX_SPIN] = MIN_SPIN
    spin3[spin3 > MAX_SPIN] = MIN_SPIN

    cbc_list1 = np.vstack((ref_lambda0,
                            ref_eta,
                            ref_spin,
                            ref_spin,
                            lambda01,
                            eta1,
                            spin1,
                            spin1)).T
    
    cbc_list2 = np.vstack((ref_lambda0,
                            ref_eta,
                            ref_spin,
                            ref_spin,
                            lambda02,
                            eta2,
                            spin2,
                            spin2)).T
    
    cbc_list3 = np.vstack((ref_lambda0,
                            ref_eta,
                            ref_spin,
                            ref_spin,
                            lambda03,
                            eta3,
                            spin3,
                            spin3)).T
    
    return cbc_list1, cbc_list2, cbc_list3

def template(param):
    lambda0, eta, chi1, chi2 = param
    mchirp = lambda0_to_mchirp(lambda0)
    m1 = mass1_from_mchirp_eta(mchirp, eta)
    m2 = mass2_from_mchirp_eta(mchirp, eta)

    tmplt, _ = get_fd_waveform(approximant=TEMPLATE,
                                    mass1=m1, mass2=m2,
                                    spin1z=chi1, spin2z=chi2,
                                    delta_f=DELTA_F, f_lower=LOW_FREQ)
    tmplt.resize(len(PSD))
    return tmplt

def match_calculation(params): 
    left_template = template(params[0:4])
    right_template = template(params[4:8])
    match, _ = pycbc_match(left_template, right_template, psd=PSD, low_frequency_cutoff=MATCH_LOW_FREQ)
    del left_template, right_template
    assert match <= 1.
    return np.concatenate((params, [match]))

def many_matches(seed):
    rng = default_rng(seed)
    parameter_list1, parameter_list2, parameter_list3 = create_cbc_list(rng, SIZE_OF_DATASET)
    parametermatch_list1 = np.array([match_calculation(params) for params in parameter_list1])
    parametermatch_list2 = np.array([match_calculation(params) for params in parameter_list2])
    parametermatch_list3 = np.array([match_calculation(params) for params in parameter_list3])
    return np.vstack((parametermatch_list1, parametermatch_list2, parametermatch_list3))

result = Parallel(n_jobs=16)(delayed(many_matches)(i) for i in range(1000))

df = pd.DataFrame(data=np.vstack(result),
                    columns=['ref_lambda0', 'ref_eta', 'ref_spin1', 'ref_spin2',
                             'lambda0', 'eta', 'spin1', 'spin2',
                             'match'])

df.to_csv(DATASET_FILE, index = False)