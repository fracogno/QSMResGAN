#!/bin/bash

rsync --exclude 'model*' -avz -e ssh s4559398@wiener.hpc.net.uq.edu.au:/scratch/cai/deepQSMGAN/ckp* .