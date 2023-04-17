#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --job-name=MassSpinNN
#SBATCH --ntasks=16
#SBATCH --time=72:00:00
#SBATCH --output=output_logfile%j
#SBATCH --error=error_logfile%j
#SBATCH --mail-user=susanna.green@port.ac.uk
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu.q

module purge
module load system
module load anaconda3/2022.10

source activate /users/sgreen/.conda/envs/PyCBCandPytorch2

echo `conda info`
echo `which python`

cd /users/sgreen/LearningMatch/LVK/Paper/MassParameters/TrainingDataset10000/
python Train.py
