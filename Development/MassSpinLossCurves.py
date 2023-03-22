import numpy as np 

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

arr = np.loadtxt("sample_data.csv", delimiter=",", dtype=float32)

#Plots the loss curve for training and validation data set
plt.figure(figsize=(8.2, 6.2))
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F', label='10000 Training Loss')
plt.semilogy(np.arange(1, len(val_loss_array)+1), val_loss_array, color='#0096FF', label='10000 Validation Loss')
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F', label='100000 Training Loss')
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F', label='100000 Training Loss')
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F', label='1000000 Training Loss')
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F', label='1000000 Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('mass_spin_loss_curve.pdf')