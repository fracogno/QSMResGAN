#!/bin/bash

rsync --exclude 'model*' -avz -e ssh uqfcogno@wiener.hpc.net.uq.edu.au:/scratch/cai/QSMResGAN/ckp* .