from Model import NeuralNetwork
from Dataset import MyDataset

from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import logging

import torch
import torch.utils.benchmark as benchmark


import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('LearningMatchGPUTimings.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/LearningMatch/Paper/EqualAlignedSpinTimings/'

#Define location of the test dataset
TEST_DATASET_FILE_PATH = DATA_DIR + '100000LambdaEtaAlignedSpinTestDataset+150000DiffusedLambdaEtaAlignedSpinTestDataset.csv'

#Define location of the trained LearningMatch model 
LEARNINGMATCH_MODEL =  DATA_DIR+'LearningMatchModel.pth'

def my_collate_fn(data):
    return tuple(data)

def run_batch_inference(compiled_model, x):
    compiled_model(x)
    
#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")    

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

x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

xy_test = MyDataset(x_test, y_test)

#Upload the already trained weights and bias
logger.info("Loading the LearningMatch model")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(LEARNINGMATCH_MODEL, map_location=device))
model.eval()

torch._dynamo.reset()
compiled_model = torch.compile(model)

batch_size_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

#mean_time_list = []
mean_time_list = torch.zeros(1, dtype=torch.float64, device=device)

#batch_size_list = [100]
for bs in batch_size_list:
    logger.info(f"The size of batch is {bs}") 
    test_data_loader  = DataLoader(xy_test, collate_fn=my_collate_fn, batch_size=bs, shuffle=False)
    x, y = next(iter(test_data_loader))

    #for i in range(10):
    #    start_time = time.time()
    #    # Make predictions for this batch
    #    y_prediction = compiled_model(x)
     #   total_time = time.time() - start_time
    #    logger.info(f"The total time is of batch {total_time}") 

    torch.cuda.synchronize()

    t_compiled_model = benchmark.Timer(
    stmt='run_batch_inference(compiled_model, x)',
    setup='from __main__ import run_batch_inference',
    globals={'compiled_model': compiled_model, 'x':x})

    t_compiled_model_runs = t_compiled_model.timeit(1000)

    mean_time_list = torch.cat((mean_time_list, torch.tensor([t_compiled_model_runs.mean/bs], device=device)), dim=0)
    #mean_time_list = mean_time_list.append(t_compiled_model_runs.mean)

    #run this five times and see how consisitent it. 
    logger.info(f"The total time is of batch {t_compiled_model_runs}") 
    logger.info(f"The total time is of batch {t_compiled_model_runs.mean/bs}")

mean_time_array = torch.tensor(mean_time_list, dtype=torch.float32, device='cpu').detach().numpy() #Gets the list off the GPU and into a numpy array
mean_time_array = np.delete(mean_time_array, 0)

plt.figure(figsize=(9, 9))
plt.loglog(batch_size_list, mean_time_array, color='#a74b9cff', linestyle='None', marker = '.', markersize=12)
plt.xlabel('Batch Size')
#plt.yscale('log')
#plt.xscale('log')
plt.ylabel('Time (seconds)')
# Setting the number of ticks 
plt.locator_params(axis='y', numticks=4) 
plt.savefig('Batch_curve.png', dpi=300, bbox_inches='tight')