### Accelerating Spike Propagation for GPU-based Spiking Neural Network Simulations

### The reference code is taken from the state-of-the-art (SOTA) implementataion:
### SOTA : "Dynamic parallelism for synaptic updating in GPU-accelerated spiking neural network simulations"
### Paper: https://www.sciencedirect.com/science/article/pii/S0925231218304168
### Code: https://bitbucket.org/bkasap/dynamicparallelismsnn/src/master/

### This repository includes SOTA (AP, N, S) algorithms and modified (AB, SKL) algorithms.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, argparse, csv
import re

# Set the number of neurons and synapses 
N    = 2500
Nsyn = 1000


#create files for neuron configurations
os.system("nvcc -I/usr/local/cuda/samples/common/inc/ -I/usr/local/cuda/include/ gpu4snn_N_NS_files.cu -o gpu4snn_N_NS_files")
os.system("././gpu4snn_N_NS_files %s %s"%(N, Nsyn))

os.system("make clean")
os.system("make rmdata")
os.system("make all")

# Read in command-line parameters
nvidia_gpu = sys.argv[1] # 'Quadro'  # GPU name
iterations = sys.argv[2] # '2000'    # number of iterations

# Create a folder with the GPU's name for the results
try: 
    os.system("mkdir %s"%(nvidia_gpu))
except FileExistsError: 
    pass

# Run the cuda code: 
os.system("./gpu4snn %s %s %s"%(N, Nsyn, iterations))

### Move the results from the `Results` directory to a dedicated path
path = './' + nvidia_gpu + '/' + iterations
os.system("mkdir %s"%(path))
os.system("mv ./Results %s"%(path))
