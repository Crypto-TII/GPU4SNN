### Accelerating Spike Propagation for GPU-based Spiking Neural Network Simulations

### The reference code is taken from the state-of-the-art (SOTA) implementataion:
### SOTA : "Dynamic parallelism for synaptic updating in GPU-accelerated spiking neural network simulations"
### Paper: https://www.sciencedirect.com/science/article/pii/S0925231218304168
### Code: https://bitbucket.org/bkasap/dynamicparallelismsnn/src/master/

### This repository includes SOTA (AP, N, S) algorithms and modified (AB, SKL) algorithms.

INCFILES=-I/usr/local/cuda/samples/common/inc/ -I/usr/local/cuda/include/ 
gpu4snn: gpu4snn.o
ARCH=-gencode arch=compute_75,code=sm_75
NVCC=nvcc -ccbin g++ -arch=sm_52
gpu4snn.o: gpu4snn.cu
	$(NVCC) $(INCFILES) -G -g -O0 $(ARCH)  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	$(NVCC) $(INCFILES) -G -g -O0 --compile --relocatable-device-code=true $(ARCH) -x cu -o  "$@" "$<"
gpu4snn: gpu4snn.o
	$(NVCC) $(INCFILES) --cudart static --relocatable-device-code=true $(ARCH) -link -o  gpu4snn  gpu4snn.o
all: gpu4snn
clean:
	rm *.o *.d gpu4snn
rmdata:
	rm -rf Results
