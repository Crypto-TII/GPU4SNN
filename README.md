# GPU4SNN
# Accelerating Spike Propagation for GPU-based Spiking Neural Network Simulations

The reference code is taken from the state-of-the-art (SOTA) implementataion:

SOTA : "Dynamic parallelism for synaptic updating in GPU-accelerated spiking neural network simulations"

Paper: https://www.sciencedirect.com/science/article/pii/S0925231218304168

Code: https://bitbucket.org/bkasap/dynamicparallelismsnn/src/master/

This repository includes SOTA (AP, N, S) algorithms and modified (AB, SKL) algorithms.

The comparison grpahs can be obtained in the "Results" directory using following commands. 

CUDA implementation of randomly connected pulse-coupled Izhikevich neurons (Izhikevich, 2003). 

Dynamic parallelism is used by AP algorithm at the synaptic update step in the spiking neural network simulations.

Cooperative Groups are used for Inter Block GPU Synchronization by AB and SKL algorithms. This is supported by CUDA Toolkit 9.0 and later versions.

Code is developed under Ubuntu with Quadro and Titan GPUs, CUDA Toolkit 11.0.

Note that dynamic parallelism requires compute capability 3.5 and separate compilation.

#### Commands to execute ###

`RUN -------------- > python gpu4snn.py $GPU_NAME$ $NUMBER_OF_ITERATIONS$`

FOR EXAMPLE ----------- > python gpu4snn.py Quadro 2000

RESULTS   ------ > ./$GPU_NAME$/$NUMBER_OF_ITERATIONS$/Results

Change Neurons (N) and Synapses (Nsyn) from gpu4snn.py file 
N = 2500
Nsyn = 1000
