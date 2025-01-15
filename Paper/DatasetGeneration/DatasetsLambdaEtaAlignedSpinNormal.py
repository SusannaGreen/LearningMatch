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
DATA_DIR = '/users/sgreen/LearningMatch/Paper/'

#Define the dataset training, validation and test dataset
DATASET_FILE = DATA_DIR+'10000LambdaEtaAlignedSpinTrainingDataset.csv'

MIN_MCHIRP = 5.0
MAX_MCHIRP = 20.0
REF_LAMBDA = 5.0
MIN_LAMBDA = (MAX_MCHIRP/REF_LAMBDA)**(-5/3)
MAX_LAMBDA = (MIN_MCHIRP/REF_LAMBDA)**(-5/3)
MIN_ETA = 0.1
MAX_ETA = 0.249999
MIN_SPIN = -0.99
MAX_SPIN = 0.99

#Define the detector
LOW_FREQ = 15 #frequency cut-off for the GW detector
MATCH_LOW_FREQ = 18
SAMPLE_RATE = 2048 #sampling rate of desired detector
TLEN = 32
DELTA_F = 1.0 / TLEN
PSD = aLIGO140MpcT1800545(1+TLEN*SAMPLE_RATE//2, delta_f=DELTA_F, low_freq_cutoff=LOW_FREQ) 

#Define the template you want LearningMatch to learn
TEMPLATE = 'IMRPhenomXAS'

SIZE_OF_DATASET = 10

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
    spin = draw_spin(rng, n_sims)
    spin2 = draw_spin(rng, n_sims)

    cbc_list = np.vstack((draw_lambda(rng, n_sims),
                            draw_eta(rng, n_sims),
                            spin,
                            spin,
                            draw_lambda(rng, n_sims),
                            draw_eta(rng, n_sims),
                            spin2,
                            spin2)).T
    
    return cbc_list

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
    parameter_list = create_cbc_list(rng, SIZE_OF_DATASET)
    return np.array([match_calculation(params) for params in parameter_list])

result = Parallel(n_jobs=16)(delayed(many_matches)(i) for i in range(1000))

df = pd.DataFrame(data=np.vstack(result),
                    columns=['ref_lambda0', 'ref_eta', 'ref_spin1', 'ref_spin2',
                             'lambda0', 'eta', 'spin1', 'spin2',
                             'match'])

df.to_csv(DATASET_FILE, index = False)