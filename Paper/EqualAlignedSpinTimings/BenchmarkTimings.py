from Model import NeuralNetwork
from Dataset import MyDataset

from torch.utils.data.dataset import Tensor
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import torch
import torch.utils.benchmark as benchmark

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/LearningMatch/Paper/EqualAlignedSpinTimings/'

#Define location of the test dataset
TEST_DATASET_FILE_PATH = DATA_DIR + '100000LambdaEtaAlignedSpinTestDataset+150000DiffusedLambdaEtaAlignedSpinTestDataset.csv'

#Define location of the trained LearningMatch model 
LEARNINGMATCH_MODEL =  DATA_DIR+'LearningMatchModel.pth'

def my_collate_fn(data):
    return tuple(data)

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"

#Data Prep
test_dataset = pd.read_csv(TEST_DATASET_FILE_PATH)

x_test = np.vstack((test_dataset.ref_lambda0.values, test_dataset.ref_eta.values,
                    test_dataset.ref_spin1.values, test_dataset.ref_spin2.values, 
                    test_dataset.lambda0.values, test_dataset.eta.values,
                    test_dataset.spin1.values, test_dataset.spin2.values)).T
y_test = test_dataset.match.values.reshape(-1,1)

x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

xy_test = MyDataset(x_test, y_test)

test_data_loader  = DataLoader(xy_test, collate_fn=my_collate_fn, batch_size=1024, shuffle=False)
x, y = next(iter(test_data_loader))

#Model Prep
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(LEARNINGMATCH_MODEL, map_location=device))
model.eval()

torch._dynamo.reset()
compiled_model = torch.compile(model)

# Make predictions for this batch
y_prediction = compiled_model(x)
torch.cuda.synchronize()

def run_batch_inference(compiled_model, x):
    compiled_model(x)
    
batch = 16

t_compiled_model = benchmark.Timer(
    stmt='run_batch_inference(compiled_model, x)',
    setup='from __main__ import run_batch_inference',
    globals={'compiled_model': compiled_model, 'x':x})

t_compiled_model_runs = t_compiled_model.timeit(1000)

print(t_compiled_model_runs)