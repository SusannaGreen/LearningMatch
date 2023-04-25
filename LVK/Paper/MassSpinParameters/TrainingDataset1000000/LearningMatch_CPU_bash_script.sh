#!/bin/bash

#SBATCH --nodes=6
#SBATCH --job-name=pycbc
#SBATCH -p sciama2.q
#SBATCH --ntasks=16
#SBATCH --time=72:00:00
#SBATCH --output=output_logfile%j
#SBATCH --error=error_logfile%j
#SBATCH --mail-user=susanna.green@port.ac.uk
#SBATCH --mail-type=FAIL

module purge
module load system
module load anaconda3/2022.10

source activate /mnt/lustre/shared_conda/envs/sgreen/PyCBCandPytorch2


echo `conda info`
echo `which python`

cd /users/sgreen/LearningMatch/LVK/Paper/MassSpinParameters/TrainingDataset1000000/
python Datasets.py
