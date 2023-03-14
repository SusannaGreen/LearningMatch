import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import logging

from sklearn import preprocessing
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

#Defining functions
def to_cpu_np(x):
    return x.cpu().detach().numpy()

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#Upload the data
TemplateBank = pd.read_csv(r'100000MassTrainingBank.csv')

#Split the data
TrainingBank, TestValidationBank = model_selection.train_test_split(TemplateBank, test_size=0.1)
TestBank, ValidationBank = model_selection.train_test_split(TestValidationBank, test_size=0.5)
print(f'The data size of the training, validation and test dataset is {len(TrainingBank), len(ValidationBank), len(TestBank)}, respectively')

#Using Standard.scalar() to re-scale the Training Bank and applying to the validation and test bank. 
scaler = preprocessing.StandardScaler()
TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.fit_transform(TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])
ValidationBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.transform(ValidationBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])
TestBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.transform(TestBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])

#Splitting into input (i.e. the template parameters) and output (the match)
x_train = np.vstack((TrainingBank.ref_mass1.values, TrainingBank.ref_mass2.values, 
TrainingBank.mass1.values, TrainingBank.mass2.values)).T
y_train = TrainingBank.match.values

x_val = np.vstack((ValidationBank.ref_mass1.values, ValidationBank.ref_mass2.values, 
ValidationBank.mass1.values, ValidationBank.mass2.values)).T
y_val = ValidationBank.match.values

x_test = np.vstack((TestBank.ref_mass1.values, TestBank.ref_mass2.values, 
TestBank.mass1.values, TestBank.mass2.values)).T
y_test = TestBank.match.values
x_real = scaler.inverse_transform(x_test)

#Convert a numpy array to a Tensor
x_train = torch.tensor(x_train, dtype=torch.float32, device='cuda')
y_train = torch.tensor(y_train, dtype=torch.float32, device='cuda')
x_val = torch.tensor(x_val, dtype=torch.float32, device='cuda')
y_val = torch.tensor(y_val, dtype=torch.float32, device='cuda')
x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear0 = torch.nn.Linear(4, 175)
        self.linear1 = torch.nn.Linear(175, 97)
        self.linear2 = torch.nn.Linear(97, 46)
        self.linear_out = torch.nn.Linear(46, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear0(x))
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear_out(x)
        return x

our_model = NeuralNetwork().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.001)
scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.2) #gamma relative to lr = 0.001

start_time = time.time()
loss_list = []
val_list = []
for epoch in range(20000): #10000 and 0.001
    #Compute the validation loss
    test_pred_y = our_model(x_test)
    val_loss = criterion(test_pred_y[:, 0], y_test)
    
    # Forward pass: Compute predicted y by passing x to the model
    pred_y = our_model(x_train)
    # Compute and print loss
    loss = criterion(pred_y[:, 0], y_train)
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #This step to is determine whether the scheduler needs to be used
    scheduler.step()
    
    #Create a loss_curve
    loss_list.append(loss.item())
    val_list.append(val_loss.item())

    #View performance during training
    if epoch%1000==0:
        print('epoch {}, training loss {}, validation loss {}'.format(epoch, loss.item(), val_loss.item())) 

print("Time taken", time.time() - start_time)

#Plots the loss curve for training and validation data set
plt.figure()
plt.semilogy(np.arange(1, len(loss_list)+1), loss_list, color='#5B2C6F')
plt.semilogy(np.arange(1, len(val_list)+1), val_list, color='#0096FF')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('loss_curve_10.pdf')


#Time taken to predict the match on the test dataset  
with torch.no_grad():
    pred_start_time = time.time()
    y_prediction = our_model(x_test) 
    torch.cuda.synchronize()
    end_time = time.time()

print("Total time taken", end_time - pred_start_time)
print("Average time taken to predict the match", (end_time - pred_start_time)/len(x_test))

#Plots the distribution of errors for the test data
error = to_cpu_np(y_prediction[:, 0] - y_test)

#Determine the values it is failing
error_list = []
for x, y in zip(error, x_real):
    if abs(x) > 0.2:
        error_list.append(error)
        print(f'The error is {x} and the parameters are {y}')
    else:
        pass

plt.figure()
plt.hist(error, bins=50, color='#5B2C6F')
plt.xlabel('Error')
plt.ylabel('Count')
plt.savefig('error_10.pdf', dpi=300)

#Creates plots to compare the actual and predicted matches 
plt.figure()
plt.scatter(to_cpu_np(y_test), to_cpu_np(y_prediction[:, 0]), color='#5B2C6F')
plt.xlabel('Actual match')
plt.ylabel('Predicted match')
plt.savefig('test_prediction_10.pdf', dpi=300)

plt.hist2d(to_cpu_np(y_test), to_cpu_np(y_prediction[:, 0]))
plt.xlabel('Actual match')
plt.ylabel('Predicted match')
plt.savefig('test_prediction_hist_10.pdf', dpi=300)

