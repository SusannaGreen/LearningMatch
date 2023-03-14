import numpy as np
import pandas as pd
import time

from pycbc.filter import matchedfilter
from pycbc.psd.analytical import aLIGOaLIGO175MpcT1800545
from pycbc.waveform import get_fd_waveform, get_td_waveform

#Defining the PSD
sample_rate = 4096
tlen = 128
delta_f = 1.0 / tlen
psd = aLIGOaLIGO175MpcT1800545(1+tlen*sample_rate//2, delta_f=delta_f, low_freq_cutoff=12)

#Training dataset
training_mass = np.random.uniform(2.0, 100, size=(100000,2))
training_reference_mass = np.random.uniform(2.0, 100, size=(100000,2))

training_parameters_list = []
match_time = []
start_time = time.time()
for ref_m1, ref_m2, m1, m2 in zip(training_reference_mass[:, 0], training_reference_mass[:, 1], training_mass[:, 0], training_mass[:, 1]):
    template_generation = time.time()
    template_reference, _ = get_fd_waveform(approximant='IMRPhenomXAS', mass1=ref_m1, mass2=ref_m2, delta_f=delta_f, f_lower=12)
    template, _ = get_fd_waveform(approximant='IMRPhenomXAS', mass1=m1, mass2=m2, delta_f=delta_f, f_lower=12)
    template_reference.resize(len(psd))
    template.resize(len(psd))
    match, Index = template.match(template_reference, psd=psd, low_frequency_cutoff=15)
    match_time.append(time.time()-template_generation)
    training_parameters_list.append([ref_m1, ref_m2, m1, m2, match])

TrainingBank =  pd.DataFrame(data=(training_parameters_list), columns=['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'match'])
TrainingBank.to_csv('MassBank.csv', index = False)

print("Time taken", time.time() - start_time)
print("Time taken to gen templates", sum(match_time))
print("Avg time per template gen", sum(match_time)/len(match_time))
