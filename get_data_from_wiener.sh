#!/bin/bash

#rsync --exclude 'model*' -avz -e ssh uqfcogno@wiener.hpc.net.uq.edu.au:/scratch/cai/QSMResGAN/ckp* .
rsync --exclude 'model*' -avz -e ssh uqfcogno@wiener.hpc.net.uq.edu.au:/scratch/cai/QSMResGAN/ckp_2020* . --exclude '*.data*'