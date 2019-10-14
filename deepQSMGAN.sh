#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=deepQSMGAN
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=40000
#SBATCH -o tensor_out.txt
#SBATCH -e tensor_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1

module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /opt/ohpc/pub/apps/tensorflow_1.13

#srun -n 1 python /scratch/cai/deepQSM_Wiener/makeSimData.py
#srun -n 1 python /scratch/cai/deepQSMGAN/trainUNET.py
#srun -n 1 python /scratch/cai/deepQSMGAN/train.py
srun -n 1 python /scratch/cai/deepQSMGAN/predict.py
