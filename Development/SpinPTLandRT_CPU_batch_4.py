import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn import preprocessing, model_selection

import torch
from torch import optim, nn, utils, Tensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from ray import tune, air
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

TemplateBank = pd.read_csv(r'100000MassSpinTrainingDataset.csv')

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
TrainingBank.ref_spin1.values, TrainingBank.ref_spin2.values, 
TrainingBank.mass1.values, TrainingBank.mass2.values,
TrainingBank.spin1.values, TrainingBank.spin2.values)).T
y_train = TrainingBank.match.values

x_val = np.vstack((ValidationBank.ref_mass1.values, ValidationBank.ref_mass2.values,
ValidationBank.ref_spin1.values, ValidationBank.ref_spin2.values,
ValidationBank.mass1.values, ValidationBank.mass2.values,
ValidationBank.spin1.values, ValidationBank.spin2.values)).T
y_val = ValidationBank.match.values

x_test = np.vstack((TestBank.ref_mass1.values, TestBank.ref_mass2.values,
TestBank.ref_spin1.values, TestBank.ref_spin2.values,
TestBank.mass1.values, TestBank.mass2.values,
TestBank.spin1.values, TestBank.spin2.values)).T
y_test = TestBank.match.values
#x_real = scaler.inverse_transform(x_test)

#Convert a numpy array to a Tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

#Convert data into trainloader
xy_train = torch.utils.data.TensorDataset(x_train, y_train)
xy_val = torch.utils.data.TensorDataset(x_val, y_val)

train_loader = torch.utils.data.DataLoader(xy_train, batch_size=64)
val_loader = torch.utils.data.DataLoader(xy_val, batch_size=64)

class LitNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.layer_3_size = config["layer_3_size"]
        self.layer_4_size = config["layer_4_size"]

        self.layer_1 = torch.nn.Linear(8, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, self.layer_3_size)
        self.layer_4 = torch.nn.Linear(self.layer_3_size, self.layer_4_size)
        self.layer_5 = torch.nn.Linear(self.layer_4_size, 1)

    def forward(self, x):
        batch_size, _ = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.relu(x)

        x = self.layer_4(x)
        x = torch.sigmoid(x)

        x = self.layer_5(x)
        return x
        
    def training_step(self, train_batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = train_batch
        x = x.view(x.size(0), -1)
        yhat = self.forward(x)
        loss = torch.nn.functional.mse_loss(yhat, y)
        # Logging to TensorBoard by default
        self.log("ptl/train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        return optimizer
    
    def validation_step(self, val_batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = val_batch
        x = x.view(x.size(0), -1)
        yhat = self.forward(x)
        loss = torch.nn.functional.mse_loss(yhat, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        return {"ptl/val_loss": avg_loss}

#Define the logger
logger = CSVLogger("logs", name="cpu_trail_1")

#Check that Pytorch recognises there is a GPU available
device = "cpu"
print(f"Using {device} device")

def train_tune(config, num_epochs=100):
    
    lit_nn = LitNN(config)
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        accelerator='cpu',
        logger=True,
        enable_progress_bar=False
        )
    trainer.fit(model=lit_nn, train_dataloaders=train_loader, val_dataloaders=val_loader)
    session.report({"loss":float(trainer.callback_metrics["ptl/val_loss"])})

def tune_asha(num_samples=100, num_epochs=200):
    config = {
        "layer_1_size": tune.randint(0,400),
        "layer_2_size": tune.randint(0,400),
        "layer_3_size": tune.randint(0,400),
        "layer_4_size": tune.randint(0,400)
    }

    scheduler = ASHAScheduler(
    max_t=num_epochs,
    grace_period=2,
    reduction_factor=2)
    
    reporter = CLIReporter(
    parameter_columns=["layer_1_size", "layer_2_size", "layer_3_size", "layer_4_size"],
    metric_columns=["loss", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_tune, num_epochs=num_epochs) 

    resources_per_trial = {"cpu": 1, "gpu": 0}

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_asha",
            progress_reporter=reporter,
            verbose=0
        ),
        param_space=config
    )
    results = tuner.fit()
    print('DONE')
    print("Best hyperparameters found were: ", results.get_best_result(filter_nan_and_inf = False).config)

tune_asha()
print('Completed')

#scontrol show job (ID)
#Exit code 0 is run while 1 is that errored
#sacct 
#conda innit bash 
#except statements for debugging: 
#import traceback
#except:
#   print('oh no!')
#   print(traceback.format_exc()) 