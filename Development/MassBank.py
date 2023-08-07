import numpy as np
import pandas as pd
import time

from pycbc.filter import matchedfilter
from pycbc.psd.analytical import aLIGOaLIGO175MpcT1800545
from pycbc.waveform import get_fd_waveform

SIZE = 10000
LOW_FREQ = 12

#Defining the PSD
sample_rate = 4096
tlen = 128
delta_f = 1.0 / tlen
psd = aLIGOaLIGO175MpcT1800545(1+tlen*sample_rate//2, delta_f=delta_f, low_freq_cutoff=LOW_FREQ)

#Training dataset
mass = np.random.uniform(3.0, 100, size=(SIZE,2))
reference_mass = np.random.uniform(3.0, 100, size=(SIZE,2))

parameters_list = []
match_time = []
start_time = time.time()
for ref_m1, ref_m2, m1, m2 in zip(reference_mass[:, 0], reference_mass[:, 1], mass[:, 0], mass[:, 1]):
    template_generation = time.time()
    template_reference, _ = get_fd_waveform(approximant='IMRPhenomXAS', mass1=ref_m1, mass2=ref_m2, delta_f=delta_f, f_lower=LOW_FREQ)
    template, _ = get_fd_waveform(approximant='IMRPhenomXAS', mass1=m1, mass2=m2, delta_f=delta_f, f_lower=LOW_FREQ)
    template_reference.resize(len(psd))
    template.resize(len(psd))
    match, Index = template.match(template_reference, psd=psd, low_frequency_cutoff=15)
    match_time.append(time.time()-template_generation)
    parameters_list.append([ref_m1, ref_m2, m1, m2, match])

MassMatchDataset =  pd.DataFrame(data=(parameters_list), columns=['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'match'])
MassMatchDataset.to_csv('10000MassMatchDataset.csv', index = False)

print("The size of the dataset", SIZE)
print("This generated a dataset called: 10000MassMatchDataset")
print(MassMatchDataset.head())
print("Time taken to generate this template bank", time.time() - start_time)
print("Total time taken to calculate all the match values", sum(match_time))
print("The average time taken to calculate the match", sum(match_time)/len(match_time))
