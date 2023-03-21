import os
import numpy as np
import pandas as pd
import time
import seaborn as sns
import logging

from sklearn import preprocessing
from sklearn import model_selection

from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
layout = {
    "ABCDE": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
        "accuracy": ["Multiline", ["accuracy/train", "accuracy/validation"]],
    },
}
writer.add_custom_scalars(layout)

#Defining functions
def to_cpu_np(x):
    return x.cpu().detach().numpy()

def scaling_the_mass(mass):
    scaled_mass = (mass - 51)/ np.sqrt(799)
    return scaled_mass

def rescaling_the_mass(mass):
    rescaled_mass = mass* np.sqrt(799) + 51
    return rescaled_mass

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#Upload the data
TemplateBank = pd.read_csv(r'Unsorted1000000MassSpinTemplateBank.csv')

#Split the data
TrainingBank, TestValidationBank = model_selection.train_test_split(TemplateBank, test_size=0.1)
TestBank, ValidationBank = model_selection.train_test_split(TestValidationBank, test_size=0.5)
print(f'The data size of the training, validation and test dataset is {len(TrainingBank), len(ValidationBank), len(TestBank)}, respectively')

#Using Standard.scalar() to re-scale the Training Bank and applying to the validation and test bank. 
scaler = preprocessing.StandardScaler()
TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'ref_spin1', 'ref_spin2', 'spin1', 'spin2']] = scaler.fit_transform(TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'ref_spin1', 'ref_spin2', 'spin1', 'spin2']])
ValidationBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'ref_spin1', 'ref_spin2', 'spin1', 'spin2']] = scaler.transform(ValidationBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'ref_spin1', 'ref_spin2', 'spin1', 'spin2']])
TestBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'ref_spin1', 'ref_spin2', 'spin1', 'spin2']] = scaler.transform(TestBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2', 'ref_spin1', 'ref_spin2', 'spin1', 'spin2']])

#Splitting into input (i.e. the template parameters) and output (the match)
x_train = np.vstack((TrainingBank.ref_mass1.values, TrainingBank.ref_mass2.values, 
TrainingBank.mass1.values, TrainingBank.mass2.values,
TrainingBank.ref_spin1.values, TrainingBank.ref_spin2.values,
TrainingBank.spin1.values, TrainingBank.spin2.values)).T
y_train = TrainingBank.match.values

x_val = np.vstack((ValidationBank.ref_mass1.values, ValidationBank.ref_mass2.values, 
ValidationBank.mass1.values, ValidationBank.mass2.values,
ValidationBank.ref_spin1.values, ValidationBank.ref_spin2.values,
ValidationBank.spin1.values, ValidationBank.spin2.values)).T
y_val = ValidationBank.match.values

x_test = np.vstack((TestBank.ref_mass1.values, TestBank.ref_mass2.values, 
TestBank.mass1.values, TestBank.mass2.values,
TestBank.ref_spin1.values, TestBank.ref_spin2.values,
TestBank.spin1.values, TestBank.spin2.values)).T
y_test = TestBank.match.values
#x_real = scaler.inverse_transform(x_test)

#Convert a numpy array to a Tensor
x_train = torch.tensor(x_train, dtype=torch.float32, device='cuda')
y_train = torch.tensor(y_train, dtype=torch.float32, device='cuda')
x_val = torch.tensor(x_val, dtype=torch.float32, device='cuda')
y_val = torch.tensor(y_val, dtype=torch.float32, device='cuda')
x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

#Convert data into trainloader
xy_train = TensorDataset(x_train, y_train)
xy_val = TensorDataset(x_val, y_val)
xy_test = TensorDataset(x_test, y_test)

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear0 = torch.nn.Linear(8, 341)
        self.linear1 = torch.nn.Linear(341, 65)
        self.linear2 = torch.nn.Linear(65, 342)
        self.linear3 = torch.nn.Linear(342, 74)
        self.linear_out = torch.nn.Linear(74, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear0(x))
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = self.linear_out(x)
        return x

# RUN ITTTT
start_time = time.time()
loss_list = []
val_list = []

epoch_number = 0
EPOCHS = 300
batch_size = 64
training_loader  = DataLoader(xy_train, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(xy_val, batch_size=batch_size, drop_last=True)

our_model = NeuralNetwork().to(device)
criterion = torch.nn.MSELoss(reduction='sum') # return the sum so we can calculate the mse of the epoch.
optimizer = torch.optim.Adam(our_model.parameters(), lr = 1e-5)#change learning rate 10e-4
scheduler = ReduceLROnPlateau(optimizer, 'min')

for epoch in range(EPOCHS):
    #print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    our_model.train(True)

    iters  = []
    epoch_loss = 0.
    val_iters = []
    vepoch_loss = 0.
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1)
        labels = labels.view(labels.size(0), -1)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = our_model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        del outputs
        del labels

        # Adjust learning weights
        optimizer.step()

        # save the current training information
        iters.append(i)
        batch_mse = loss/inputs.size(0)
        del batch_mse
        epoch_loss += float(loss) #added float to address the memory issues
    
    our_model.train(False)

    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.view(vinputs.size(0), -1)
        vlabels = vlabels.view(vlabels.size(0), -1)
        voutputs = our_model(vinputs)
        vloss = criterion(voutputs, vlabels)

        del voutputs
        del vlabels

        val_iters.append(i)
        vbatch_mse = vloss/vinputs.size(0)
        del vbatch_mse
        vepoch_loss += float(vloss)

    epoch_mse = epoch_loss/(len(iters) * inputs.size(0))
    vepoch_mse = vepoch_loss/(len(val_iters) * vinputs.size(0))
    print('EPOCH: {} LOSS train {} valid {}'.format(epoch_number, epoch_mse, vepoch_mse),end="\r")

    del epoch_loss
    del vepoch_loss
    del inputs
    del vinputs
    
    #This step to is determine whether the scheduler needs to be used 
    scheduler.step(vepoch_mse)

    #Create a loss_curve
    loss_list.append(epoch_mse)
    val_list.append(vepoch_mse)

    writer.add_scalar("loss/train", epoch_mse, epoch_number)
    writer.add_scalar("loss/validation", vepoch_mse, epoch_number)
    #writer.add_scalar("learning rate", scheduler._last_lr)
    epoch_number += 1

    del epoch_mse
    del vepoch_mse

print("Time taken", time.time() - start_time)
writer.flush()
writer.close()

scheduler.optimizer.param_groups[0]['lr']
optimizer.state

our_model.train(False)
#Time taken to predict the match on the test dataset  
with torch.no_grad():
    pred_start_time = time.time()
    y_prediction = our_model(x_test) 
    torch.cuda.synchronize()
    end_time = time.time()

print("Total time taken", end_time - pred_start_time)
print("Average time taken to predict the match", (end_time - pred_start_time)/len(x_test))

error = to_cpu_np(y_prediction[:, 0] - y_test)

#Determine the values it is failing
error_list = []
for x, y in zip(error, x_real):
    if abs(x) > 0.2:
        error_list.append(error)
        print(f'The error is {x} and the parameters are {y}')
    else:
        pass

#Save the Neural Network 
torch.save(our_model.state_dict(), '/users/sgreen/GPU/Width/1000000_mass_spin_model.pth')

#A really complicated way of getting the list off the GPU and into a numpy array
loss_array = torch.tensor(loss_list, dtype=torch.float32, device='cpu').detach().numpy()
val_loss_array = torch.tensor(val_list, dtype=torch.float32, device='cpu').detach().numpy()

np.savetxt("1000000_mass_spin_training.csv", loss_array, delimiter=",")
np.savetxt("1000000_mass_spin_validation.csv", val_loss_array, delimiter=",")

#Plots the loss curve for training and validation data set
plt.figure(figsize=(8.2, 6.2))
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#5B2C6F', label='Training Loss')
plt.semilogy(np.arange(1, len(val_loss_array)+1), val_loss_array, color='#0096FF', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('1000000_mass_spin_loss_curve_plot.pdf')

#Creates plots to compare the actual and predicted matches 
plt.figure(figsize=(8, 6))
x = to_cpu_np(y_test)
y = to_cpu_np(y_prediction[:, 0])
plt.scatter(x,y, s=2, color='#0096FF')
plt.axline((0, 0), slope=1, color='k')
plt.xlabel('Actual Match')
plt.ylabel('Predicted Match')
plt.savefig('1000000_mass_spin_actual_predicted_plot.pdf', dpi=300)

#Creates plots to compare the errors
plt.figure(figsize=(9, 7))
plt.hist(error, bins=30, range=[error.min(), error.max()], color='#5B2C6F', align='mid', label='Errors for all match values')
plt.hist(error[x > .95], bins=30, range=[error.min(), error.max()], color='#0096FF', align='mid', label='Errors for match values over 0.95')
plt.xlim([error.min(), error.max()])
plt.xticks([-0.1, -0.05, -0.01, 0.01, 0.05, 0.1])
plt.yscale('log')
plt.xlabel('Error')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.savefig('1000000_mass_spin_error_plot.pdf', dpi=300)

#Creates a Actual Match and Predicted Match plot with residuals
fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 7), sharex=True, height_ratios=[3, 1])
ax1.scatter(x,y, s=2, color='#0096FF')
ax1.axline((0, 0), slope=1, color='k')

sns.residplot(x=x, y=y, color = '#0096FF', scatter_kws={'s': 8}, line_kws={'linewidth':20})

fig.supxlabel('Actual Match')
fig.supylabel('Predicted Match')
plt.savefig('1000000_mass_spin_actual_predicted_plot_with residuals.pdf', dpi=300)