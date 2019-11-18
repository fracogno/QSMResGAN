#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=deepQSMResGAN
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=50000
#SBATCH -o out_deepQSMResGAN.txt
#SBATCH -e error_deepQSMResGAN.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1

module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /opt/ohpc/pub/apps/tensorflow_1.13

#srun -n 1 python makeSimData.py
#srun -n 1 python trainUNET.py
#srun -n 1 python train_NEW.py
srun -n 1 python predict.py
