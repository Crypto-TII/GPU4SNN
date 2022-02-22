# GPU4SNN
# Accelerating Spike Propagation for GPU-based Spiking Neural Network Simulations
_(This work has been submitted to Neural Networks and is currently under review)_

This repository includes the CUDA implementation of state-of-the-art (AP, N, S) algorithms and two new proposed (AB, SKL) algorithms for SNN simulation based on a network of randomly connected pulse-coupled Izhikevich neurons (Izhikevich, 2003). 

The reference code is taken from the state-of-the-art (SOTA) implementation:
_"Dynamic parallelism for synaptic updating in GPU-accelerated spiking neural network simulations"_  ([Paper](https://www.sciencedirect.com/science/article/pii/S0925231218304168)) ([Code](https://bitbucket.org/bkasap/dynamicparallelismsnn/src/master/)).

#### Commands to execute ###

Please note that this repository contains a demo-notebook under [GPU4SNN Demo Notebook](https://github.com/Crypto-TII/GPU4SNN/blob/main/GPU4SNN%20-%20Demo%20Notebook.ipynb).  
Alternatively, execute the following commands:  

```
RUN -------------- > python gpu4snn.py $GPU_NAME$ $NUMBER_OF_ITERATIONS$
FOR EXAMPLE ------ > python gpu4snn.py Quadro 2000
RESULTS   -------- > ./$GPU_NAME$/$NUMBER_OF_ITERATIONS$/Results
```

#### Notes ####

* The dynamic parallelism is used by AP algorithm at the synaptic update step in the spiking neural network simulations. Note that dynamic parallelism requires compute capability 3.5 and separate compilation.
* Cooperative Groups are used for Inter Block GPU Synchronization by AB and SKL algorithms. This is supported by CUDA Toolkit 9.0 and later versions.
* The code is developed under Ubuntu with Quadro and Titan GPUs, CUDA Toolkit 11.5.
* Change Neurons (N) and Synapses (Nsyn) in the `gpu4snn.py` file 
```
N    = 2500
Nsyn = 1000
```


