import numpy as np
from numpy.random import default_rng
import pandas as pd
import timeit
import logging

from pycbc.filter import match as pycbc_match
from pycbc.psd.analytical import aLIGO140MpcT1800545
from pycbc.waveform import get_fd_waveform
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta

from joblib import Parallel, delayed
import joblib

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('BatchedTimingsCPU.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory for the output
DATA_DIR = '/users/sgreen/LearningMatch/Paper/EqualAlignedSpinTimings/'

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

def many_matches(parameter_list):
    return np.array([match_calculation(params) for params in parameter_list])

batch_size_list = [100, 1000, 10000]
#n_jobs_list = [16, 64]

for batch_size in batch_size_list:
    rng = default_rng(1)
    parameter_list = create_cbc_list(rng, batch_size)
    logger.info(f'The batch_size is {batch_size}')
    
    t_matches = timeit.Timer(
    stmt='many_matches(parameter_list)',
    setup='from __main__ import many_matches',
    globals={'parameter_list':parameter_list})

    t_matches_runs = t_matches.timeit(100)

    logger.info(f"The total time is of batch {t_matches_runs}") 
#for nom_jobs in n_jobs_list:
#    logger.info(f'The number of CPUs is {nom_jobs}')
#    
#    for batch_size in batch_size_list:
#        rng = default_rng(1)
#        parameter_list = create_cbc_list(rng, batch_size)
#        logger.info(f'The batch_size is {batch_size}')
#        result = Parallel(n_jobs=nom_jobs)(delayed(many_matches)(params) for params in range(parameter_list))
