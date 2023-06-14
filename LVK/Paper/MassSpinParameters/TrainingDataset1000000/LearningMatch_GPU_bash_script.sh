#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --job-name=LearningMatch
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

source activate /mnt/lustre/shared_conda/envs/sgreen/NewPyCBCandPytorch2

echo `conda info`
echo `which python`

cd /users/sgreen/LearningMatch/LVK/Paper/MassSpinParameters/TrainingDataset1000000/
python Train.py
