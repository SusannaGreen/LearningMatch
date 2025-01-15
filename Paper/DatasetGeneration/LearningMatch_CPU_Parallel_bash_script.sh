#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=pycbc
#SBATCH -p sciama3.q 
#SBATCH --ntasks=16
#SBATCH --time=72:00:00
#SBATCH --output=output_logfile%j
#SBATCH --error=error_logfile%j
#SBATCH --mail-user=susanna.green@port.ac.uk
#SBATCH --mail-type=FAIL

module purge
module load system
module load anaconda3/2022.10

source activate /mnt/lustre2/shared_conda/envs/sgreen/PyCBCandOptuna_2

echo `conda info`
echo `which python`

cd /users/sgreen/LearningMatch/Paper/
python DatasetsLambdaEtaAlignedSpinNormal.py
