#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --job-name=LearningMatch
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=128:00:00
#SBATCH --output=output_logfile%j
#SBATCH --error=error_logfile%j
#SBATCH --mail-user=susanna.green@port.ac.uk
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu.q

module purge
module load system
module load anaconda3/2022.10

source activate /mnt/lustre2/shared_conda/envs/sgreen/XanPytorch_2/

echo `conda info`
echo `which python`

cd /users/sgreen/LearningMatch/Paper/EqualAlignedSpinDeeperModelLayers4Layers6/
python Train.py
