import numpy as np
import pandas as pd
import time

from pycbc.filter import matchedfilter
from pycbc.psd.analytical import aLIGOaLIGO175MpcT1800545
from pycbc.waveform import get_fd_waveform

SIZE = 100000 #size of the desired template bank 
LOW_FREQ = 12

#Defining the PSD
sample_rate = 4096
tlen = 128 
delta_f = 1.0 / tlen
psd = aLIGOaLIGO175MpcT1800545(1+tlen*sample_rate//2, delta_f=delta_f, low_freq_cutoff=LOW_FREQ)


#Training dataset
mass = np.random.uniform(3.0, 100, size=(SIZE, 2))
reference_mass = np.random.uniform(3.0, 100, size=(SIZE, 2))
spin = np.random.uniform(-0.99, 0.99, size=(SIZE, 2))
reference_spin = np.random.uniform(-0.99, 0.99, size=(SIZE, 2))

parameters_list = []
match_time = []
start_time = time.time()
for ref_m1, ref_m2, m1, m2, ref_s1, ref_s2, s1, s2 in zip(reference_mass[:, 0], reference_mass[:, 1], mass[:, 0], mass[:, 1], reference_spin[:, 0], reference_spin[:, 1], spin[:, 0], spin[:, 1]):
    template_generation = time.time()
    template_reference, _ = get_fd_waveform(approximant='IMRPhenomXAS', mass1=ref_m1, mass2=ref_m2, spin1z=ref_s1, spin2z=ref_s2, delta_f=delta_f, f_lower=LOW_FREQ)
    template, _ = get_fd_waveform(approximant='IMRPhenomXAS', mass1=m1, mass2=m2, spin1z=s1, spin2z=s2, delta_f=delta_f, f_lower=LOW_FREQ)
    template_reference.resize(len(psd))
    template.resize(len(psd))
    match, Index = template.match(template_reference, psd=psd, low_frequency_cutoff=15)
    match_time.append(time.time()-template_generation)
    parameters_list.append([ref_m1, ref_m2, m1, m2, ref_s1, ref_s2, s1, s2, match])

MassSpinMatchDataset =  pd.DataFrame(data=(parameters_list), columns=['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'ref_spin1', 'ref_spin2', 'spin1', 'spin2', 'match'])
MassSpinMatchDataset.to_csv('100000MassSpinMatchDataset.csv', index = False)

print("The size of the dataset", SIZE)
print("This generated a dataset called: 100000MassSpinMatchDataset")
print(MassSpinMatchDataset.head())
print("Time taken to generate this template bank", time.time() - start_time)
print("Total time taken to calculate all the match values", sum(match_time))
print("The average time taken to calculate the match", sum(match_time)/len(match_time))
#Dataset 100,000 test and 100,000 test - log(base 10) Mass
#Generate the numbers in log space and then create the file in normal masses - same different name.