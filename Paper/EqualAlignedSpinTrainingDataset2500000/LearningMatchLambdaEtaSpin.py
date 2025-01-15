#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 
import numpy as np
import pandas as pd
import time
import logging
from typing import Iterator, List, Sized

import rff

import torch
import torch.nn as nn
from torch.utils.data.dataset import Tensor
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp.grad_scaler import GradScaler

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

torch.set_float32_matmul_precision('high')

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('LambdaEtaSpin.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define the functions
def to_cpu_np(x):
    return x.cpu().detach().numpy()

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/LearningMatch/Paper/'

#Define location to the training and validation dataset
TRAINING_DATASET_FILE_PATH = DATA_DIR+r'1000000LambdaEtaAlignedTrainingDatasetNewRange+1500000DiffusedLambdaEtaSpinTrainingDataset9AlignedNewRange.csv'
VALIDATION_DATASET_FILE_PATH = DATA_DIR+r'100000LambdaEtaAlignedValidationDatasetNewRange+150000DiffusedLambdaEtaSpinValidationDataset9AlignedNewRange.csv'
TEST_DATASET_FILE_PATH = DATA_DIR+r'100000LambdaEtaAlignedTestDatasetNewRange+150000DiffusedLambdaEtaSpinTestDataset9AlignedNewRange.csv'

#Define output location of the LearningMatch model
LEARNINGMATCH_MODEL = DATA_DIR+'LearningMatchModelLambdaEtaSpinLR6Batch1024.pth'

#Defining the location of the outputs
LOSS_CURVE = DATA_DIR+'LossCurveLambdaEtaSpinDiffusionLR6Batch1024.png'
ERROR_HISTOGRAM = DATA_DIR+'ErrorLambdaEtaSpinDiffusionLR6Batch1024.png'
ACTUAL_PREDICTED_PLOT = DATA_DIR+'ActualPredictLambdaEtapinDiffusionLR6Batch1024.png'

#Define values for the LearningMatch model
EPOCHS = 2000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-6

#Define the Model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.embed = nn.Sequential(
              nn.Linear(4, 1024), 
              nn.ReLU(), 
              nn.Linear(1024, 1024),
              nn.ReLU(), 
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 16))
        self.crunch = nn.Sequential(
              nn.Linear(4+16, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1024), 
              nn.ReLU(),
              nn.Linear(1024, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.embed(x[..., 0:4])
        b = self.embed(x[..., 4:8])
        diffsq = torch.pow(torch.sub(a, b), 2)
        avg = torch.div(torch.add(x[..., 0:4],x[..., 4:8]), 2)
        return self.crunch(torch.concat((avg, diffsq), dim=1))

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")

#Reading the Training, validation and test dataset
logger.info("Reading in the data")
TrainingBank = pd.read_csv(TRAINING_DATASET_FILE_PATH)
ValidationBank = pd.read_csv(VALIDATION_DATASET_FILE_PATH)
logger.info(f'The data size of the training and validation dataset is {len(TrainingBank), len(ValidationBank)}, respectively')

#Splitting into input (i.e. the template parameters) and output (the match)
x_train = np.vstack((TrainingBank.ref_lambda0.values, TrainingBank.ref_eta.values, 
                     TrainingBank.ref_spin1.values, TrainingBank.ref_spin2.values, 
                     TrainingBank.lambda0.values, TrainingBank.eta.values,
                     TrainingBank.spin1.values, TrainingBank.spin2.values)).T
y_train = TrainingBank.match.values.reshape(-1,1)

x_val = np.vstack((ValidationBank.ref_lambda0.values, ValidationBank.ref_eta.values,
                   ValidationBank.ref_spin1.values, ValidationBank.ref_spin2.values, 
                   ValidationBank.lambda0.values, ValidationBank.eta.values,
                   ValidationBank.spin1.values, ValidationBank.spin2.values)).T
y_val = ValidationBank.match.values.reshape(-1,1)

#Convert a numpy array to a Tensor
logger.info("Converting to datasets to a trainloader")
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)#.reshape(1, *y_train.shape).T
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)#.reshape(1, *y_val.shape).T

def my_collate_fn(data):
    return tuple(data)

class MyDataset(TensorDataset):
    def __init__(self, *tensors: Tensor) -> None:
        super().__init__()
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitems__(self, indices):
        return tuple(tensor[indices] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

#Convert data into TensorDataset
xy_train = MyDataset(x_train, y_train)
xy_val = MyDataset(x_val, y_val)

del x_train
del y_train
del x_val
del y_val

class FastRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized, generator=None) -> None:
        self.data_source = data_source
        self.generator = generator or torch.Generator()

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        rand_indices = torch.randperm(n, generator=self.generator).tolist()
        return iter(rand_indices)

    def __len__(self) -> int:
        return len(self.data_source)

class FastBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(iter(self.sampler))
        if self.drop_last:
            indices = indices[:len(indices) - len(indices) % self.batch_size]
        return (indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


sampler = FastBatchSampler(FastRandomSampler(xy_train), BATCH_SIZE, drop_last=True)
vsampler = FastBatchSampler(FastRandomSampler(xy_val), BATCH_SIZE, drop_last=True)

#Convert data into DataLoaders
training_loader  = DataLoader(xy_train, collate_fn=my_collate_fn, sampler=sampler, num_workers=0, pin_memory=True)
validation_loader = DataLoader(xy_val, collate_fn=my_collate_fn, sampler=vsampler, num_workers=0, pin_memory=True)

logger.info("Uploading the model")
our_model = NeuralNetwork().to(device)
criterion = torch.nn.MSELoss().to(device) # return the sum so we can calculate the mse of the epoch.
optimizer = torch.optim.Adam(our_model.parameters(), lr = LEARNING_RATE)
#scheduler = ReduceLROnPlateau(optimizer, 'min')
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(training_loader), epochs=10)
compiled_model = torch.compile(our_model)

logger.info("Model successfully loaded")

for parameter in our_model.parameters():
    logger.info(len(parameter))

trainable_params =sum(p.numel()for p in our_model.parameters()if p.requires_grad)
logger.info(f'The number of trainable parameters is {trainable_params}')

start_time = time.time()

loss_list = torch.zeros(1, dtype=torch.float64, device=device)
val_list = torch.zeros(1, dtype=torch.float64, device=device)

nbatches = len(training_loader)
val_nbatches = len(validation_loader)

epoch_number = 0
for epoch in range(EPOCHS):
    # Make sure gradient tracking is on, and do a pass over the data
    compiled_model.train(True)

    iters  = torch.zeros(1, dtype=torch.float64, device=device)
    epoch_loss = torch.zeros(1, dtype=torch.float64, device=device)
    val_iters = torch.zeros(1, dtype=torch.float64, device=device)
    vepoch_loss = torch.zeros(1, dtype=torch.float64, device=device)

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        outputs = compiled_model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        del outputs
        del labels

        # save the current training information
        epoch_loss += loss.detach()
            
    compiled_model.train(False)

    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.to(device, non_blocking=True)
        vlabels = vlabels.to(device, non_blocking=True)
        voutputs = compiled_model(vinputs)
        vloss = criterion(voutputs, vlabels)

        del voutputs
        del vlabels

        vepoch_loss += vloss.detach()

    epoch_mse = epoch_loss/nbatches
    vepoch_mse = vepoch_loss/val_nbatches
    logger.info('EPOCH: {} TRAINING LOSS {} VALIDATION LOSS {}'.format(epoch_number, float(epoch_mse.detach()), float(vepoch_mse.detach())))
    
    #scheduler.step(vepoch_mse)
    
    del epoch_loss
    del vepoch_loss
    del inputs
    del vinputs

    loss_list = torch.cat((loss_list, epoch_mse.detach()), 0)
    val_list = torch.cat((val_list, vepoch_mse.detach()), 0)

    del epoch_mse
    del vepoch_mse

    epoch_number += 1

    #Checkpoint every 10 epochs
    if epoch_number % 10 == 0:
        torch.save(our_model.state_dict(), LEARNINGMATCH_MODEL)
        logger.info("Model saved at epoch %s", epoch_number)

logger.info("Time taken to train LearningMatch %s", time.time() - start_time)

#Save the trained LearningMatch model 
torch.save(our_model.state_dict(), LEARNINGMATCH_MODEL)

del training_loader
del validation_loader
del sampler
del vsampler

#Reading the test dataset
logger.info("Reading in the test dataset")
test_dataset = pd.read_csv(TEST_DATASET_FILE_PATH)
logger.info(f'The size of the test dataset is {len(test_dataset)}')

#Convert to a Tensor
logger.info("Converting the test dataset into a tensor")
x_test = np.vstack((test_dataset.ref_lambda0.values, test_dataset.ref_eta.values,
                    test_dataset.ref_spin1.values, test_dataset.ref_spin2.values, 
                    test_dataset.lambda0.values, test_dataset.eta.values,
                    test_dataset.spin1.values, test_dataset.spin2.values)).T
y_test = test_dataset.match.values.reshape(-1,1)

x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda')

#Convert data into TensorDataset
xy_test = MyDataset(x_test, y_test)

#Convert data into DataLoaders
test_data_loader  = DataLoader(xy_test, collate_fn=my_collate_fn, batch_size=1024, shuffle=False)

error_test_dataset = torch.zeros((1,1), dtype=torch.float64, device=device)
y_test_dataset = torch.zeros((1,1), dtype=torch.float64, device=device)
y_prediction_dataset = torch.zeros((1,1), dtype=torch.float64, device=device)

#Time taken to predict the match on the test dataset
logger.info("LearningMatch is predicting the Match for your dataset")
with torch.inference_mode():
    pred_start_time = time.time()
    for i, data in enumerate(test_data_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Make predictions for this batch
        y_prediction = compiled_model(inputs)
        error = labels - y_prediction
        error_test_dataset = torch.cat((error_test_dataset, error.detach()), 0)
        y_test_dataset = torch.cat((y_test_dataset, labels.detach()), 0)
        y_prediction_dataset = torch.cat((y_prediction_dataset, y_prediction.detach()), 0)
    torch.cuda.synchronize()
    end_time = time.time()
  
logger.info(("Total time taken", end_time - pred_start_time))
logger.info(("Average time taken to predict the match", (end_time - pred_start_time)/len(x_test)))

#Creates a plot that compares the actual match values with LearningMatch's predicted match values 
logger.info("Creating a plot that compares tha actual match values with predicted match values, with residuals")  

x = to_cpu_np(y_test_dataset)
y = to_cpu_np(y_prediction_dataset)

fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 9), sharex=True, height_ratios=[3, 1])
ax1.scatter(x,y, s=8, color='#4690ef')
ax1.axline((0, 0), slope=1, color='k')
ax1.set_ylabel('Predicted Match')

sns.residplot(x=x, y=y, color = '#4690ef', scatter_kws={'s': 8}, line_kws={'linewidth':20})
ax2.set_ylabel('Error')

fig.supxlabel('Actual Match')
plt.savefig(ACTUAL_PREDICTED_PLOT, dpi=300, bbox_inches='tight')

#Creates a histogram of the errors
logger.info("Creating a histogram of the errors") 

error = to_cpu_np(error_test_dataset)

plt.figure(figsize=(9, 9))
plt.hist(error, bins=30, range=[error.min(), error.max()], color='#a74b9cff', align='mid', label='Errors for all match values')
plt.hist(error[x > .95], bins=30, range=[error.min(), error.max()], color='#4690ef', align='mid', label='Errors for match values over 0.95')
plt.xlim([error.min(), error.max()])
plt.xticks([-0.05, -0.01, 0.01, 0.05])
plt.yscale('log')
plt.xlabel('Error')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.savefig(ERROR_HISTOGRAM, dpi=300, bbox_inches='tight')

#Plots the loss curve for training and validation data set
logger.info("Creating a loss curve which compares the training loss with validation loss")  

loss_array = to_cpu_np(loss_list)
val_loss_array = to_cpu_np(val_list)

loss_array = np.delete(loss_array, 0)
val_loss_array = np.delete(val_loss_array, 0)

plt.figure(figsize=(9, 9))
plt.semilogy(np.arange(1, len(loss_array)+1), loss_array, color='#a74b9cff', label='Training Loss')
plt.semilogy(np.arange(1, len(val_loss_array)+1), val_loss_array, color='#4690ef', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(LOSS_CURVE, dpi=300, bbox_inches='tight')