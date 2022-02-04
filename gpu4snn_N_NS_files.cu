/// Accelerating Spike Propagation for GPU-based Spiking Neural Network Simulations

/// The reference code is taken from the state-of-the-art (SOTA) implementataion: 
/// SOTA : "Dynamic parallelism for synaptic updating in GPU-accelerated spiking neural network simulations"
/// Paper: https://www.sciencedirect.com/science/article/pii/S0925231218304168
/// Code: https://bitbucket.org/bkasap/dynamicparallelismsnn/src/master/

/// This repository includes SOTA (AP, N, S) algorithms and modified (AB, SKL) algorithms.

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <ctime>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#include <sys/stat.h>

#include <assert.h>
///#include <stdio.h>
///#include <stdlib.h>
#include <string.h>
//#include <sm_11_atomic_functions.h>

#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups; 

////////////////

using namespace std;

class Neuron
{
	public:
		float	v;
		float 	u;
		float	a;
		float	b;
		float	c;
		float	d;
		float	I;
		int		nospks;
		int		neuronid;
};

////////////////////////////// global sync here ///////////////////////////

//#define ITER_COUNT 2000

// Initialize the random states
__global__ void randgeneratorinit(unsigned int seed, curandState_t* states, int N) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if ( id < N ){
	/* we have to initialize the state */
		curand_init(seed,	/* the seed can be the same for each core, here we pass the time in from the CPU */
			  id,		/* the sequence number should be different for each core (unless you want all
							 cores to get the same sequence of numbers for some reason - use thread id! */
			  0,		/* the offset is how much extra we advance in the sequence for each call, can be 0 */
			  &states[id]);
		}
}

// Initialize the random states
__global__ void randgeneratorinit2(float* rand_float, curandState_t* state, int N) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t localState ;
	
	if ( id < N ){
	/* we have to initialize the state */
		localState = state[id];
		rand_float[id] = curand_uniform(&localState); 
	}
}

//__device__ float rand_float[1] ;  
// Initialize the neural parameters
__global__ void initNeuron(int N_exc, int N, Neuron *neuron, float* rand_float){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("%d \n", id);
	
	// initialize excitatory neuron parameters
	if ( id < N ){
	if ( id < N_exc ){
		neuron[id].a = 0.02f;
		neuron[id].b = 0.2f;
		neuron[id].c = -65.0f+(15.f*powf(rand_float[id],2.0f));
		neuron[id].d = 8.0f-(6.f*powf(rand_float[id],2.0f));
		neuron[id].v = -65.0f;
		neuron[id].u = neuron[id].v*neuron[id].b;
		neuron[id].I = 0.0f;
		neuron[id].nospks = 0;
		neuron[id].neuronid = id;
	}
	// initialize inhibitory neuron parameters
	else if (id >= N_exc and id < N ){
		neuron[id].a = 0.02f+(0.08f*rand_float[id]);
		neuron[id].b = 0.25f-(0.05f*rand_float[id]);
		neuron[id].c = -65.0f;
		neuron[id].d = 2.0f;
		neuron[id].v = -65.0f;
		neuron[id].u = neuron[id].v*neuron[id].b;
		neuron[id].I = 0.0f;
		neuron[id].nospks = 0;
		neuron[id].neuronid = id;
	}
	}
}

// Write filename
// helper functions to write simulation results in files
const char * filename(char * buffer, string varname){
	string fname;
	fname = buffer + varname;
	return fname.c_str();
}

void writeNeuronInfo(std::ofstream& output, Neuron n){
	output << n.a << " " << n.b << " " << n.c << " " << n.d << " " << n.v << " " << n.u << " " << n.I << " " << n.neuronid << " " << n.nospks << "\n";
}

// main function simulates the same network (N and S)
// for different firing regimes (mode: 0 quiet, 1 balanced, 2 irregular)
// with all the synaptic update algorithms (dynamic: 0 AP-algorithm, 1 N-algorithm, 2 S-algorithm)
#define concat(first, second) first second

int main(int argc, char **argv){
	// Number of neurons in the network: N
	// Number of synapses per neuron: N_syn
	const int N = atoi(argv[1]);
	const int N_syn = atoi(argv[2]);
	
	// (void *)&d_Neuron,  (void *)&d_spikes, (void *)&d_conn_w, (void *)&d_conn_idx, (void *)&d_Isyn, (void *)&N, (void *)&N_exc, (void *)&N_syn, (void *)&d_totalspkcount, (void *)&devStates, (void *)&mode, (void *)&d_count,};    				

	// Number of excitatory 80% and inhibitory 20% connections
	const int N_exc = ceil(4*N/5);
	const int N_inh = ceil(1*N/5);

	cout << "Number of neurons: " << N << "\nnumber of synapses per neuron: "  << N_syn << "\n";
	printf("%d Neurons in the network: %d excitatory and %d inhibitory \n", N, N_exc, N_inh);

	printf("Allocating space on GPU memory: \n");
	// Allocate space on GPU memory for neural parameters
	// for N neurons at time-steps t_n and t_(n+1)
	//Neuron *d_Neuron, *h_Neuron;
	//h_Neuron = (Neuron *)malloc(N*sizeof(Neuron));
	//cudaGetErrorString(cudaMalloc(&d_Neuron, N*sizeof(Neuron)));

	
	// for N neurons to keep spikes
	//bool *d_spikes, *h_spikes;
	//h_spikes = (bool *)malloc(N*sizeof(bool));
	//cudaGetErrorString(cudaMalloc(&d_spikes, N*sizeof(bool)));
	printf("Memory allocated for neurons\n");

	// for connectivity matrix
	float *d_conn_w; float *h_conn_w;
	h_conn_w = (float *)malloc(N_syn*N*sizeof(float));
	cudaGetErrorString(cudaMalloc(&d_conn_w, N_syn*N*sizeof(float)));
	printf("Memory allocated for connectivity matrix\n");

	// allocate memory on the GPU memory for the connectivity
	int *d_conn_idx, *h_conn_idx;
	h_conn_idx = (int *)malloc(N_syn*N*sizeof(int));
	cudaGetErrorString(cudaMalloc(&d_conn_idx, N_syn*N*sizeof(int)));

	// for synaptic input to N neurons
	float *d_Isyn;
	cudaGetErrorString(cudaMalloc(&d_Isyn, N*sizeof(float)));
	printf("Memory allocated for synapses\n");

	// gridSize and blockSize for N operations
	int blockSizeN;		// The launch configurator returned block size
	int minGridSize;	// The minimum grid size needed to achieve the
						// maximum occupancy for a full device launch
	int gridSizeN;		// The actual grid size needed, based on input size
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeN, initNeuron, 0, 0);
	// Round up according to array size
	gridSizeN = (N + blockSizeN - 1) / blockSizeN;
	
	// calculate theoretical occupancy
  	int maxActiveBlocks;
  	cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 	initNeuron, blockSizeN, 
                                                 	0);

  	int device;
  	cudaDeviceProp props;
  	cudaGetDevice(&device);
  	cudaGetDeviceProperties(&props, device);

  	float occupancy = (maxActiveBlocks * blockSizeN / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);

  	printf("Launched blocks of size %d. Theoretical occupancy: %f\n", 
         blockSizeN, occupancy);

	
	printf("blockSizeN = %d , gridSizeN = %d, minGridSize = %d  maxActiveBlocks = %d \n", blockSizeN, gridSizeN, minGridSize, maxActiveBlocks);
	// initialize random number generator to be used for stochastic input
	printf("Initializing random number generators\n");
	curandState_t *devStates;
	cudaGetErrorString(cudaMalloc((void **)&devStates, N*sizeof(curandState_t)));
	
	//int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK/32) + 1);
 	int sMemSize = 0;
  	int numBlocksPerSm = 0;
  	//int numThreads = THREADS_PER_BLOCK;  			
	
	cudaDeviceProp deviceProp;
  	int devID = findCudaDevice(argc, (const char **)argv);
  	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
	int numSms = deviceProp.multiProcessorCount;   	
	
	int minGridSize_rng, blockSizeN_rng, gridSizeN_rng;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_rng, &blockSizeN_rng, randgeneratorinit, 0, 0));
				
  	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, initNeuron, blockSizeN_rng, sMemSize));
      	gridSizeN_rng = (N + blockSizeN_rng - 1) / blockSizeN_rng;
  				
  	dim3 dimGrid_rng(gridSizeN_rng, 1, 1),
      		dimBlock_rng(blockSizeN_rng, 1, 1); //
    	occupancy = (numBlocksPerSm * blockSizeN_rng / props.warpSize) / 
                 		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      	printf("randgeneratorinit numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      				
      	printf("randgeneratorinit blockSizeN_rng = %d , maxActiveBlocks = %d gridSizeN_rng =%d \n", blockSizeN_rng, numSms * numBlocksPerSm, gridSizeN_rng);
		
	float *host_rand_float ; //=  (float *)malloc(N*sizeof(float));
	host_rand_float =  (float *)malloc(N*sizeof(float));
		
	float *dev_rand_float;
	cudaGetErrorString(cudaMalloc((void **)&dev_rand_float, N*sizeof(float)));	
					//randgeneratorinit<<<gridSizeN, blockSizeN>>>(time(NULL), devStates);
	unsigned int seed_cpu = 1643957724 ; //time(NULL);
	cout<< "seed_cpu " <<seed_cpu<<"\n";
	randgeneratorinit<<<dimGrid_rng, dimBlock_rng>>>(seed_cpu, devStates, N);
	cudaDeviceSynchronize();
	
	randgeneratorinit2<<<dimGrid_rng, dimBlock_rng>>>(dev_rand_float, devStates, N);
	cudaDeviceSynchronize();
	
	// Initialize connectivity matrix on the GPU
	//printf("Initializing random connectivity matrix values\n");
	
	cudaGetErrorString(cudaMemcpy(host_rand_float, dev_rand_float, N*sizeof(float), cudaMemcpyDeviceToHost));
	
	FILE *filePtr_N_states; 
	
	const char* q1 = "N_states";
    	const char* q2 = argv[1];

    	//char * qq = (char*) malloc((strlen(q1)+ strlen(q2))*sizeof(char));
    	
	
	char* filename =  (char*) malloc((strlen(q1)+ strlen(q2))*sizeof(char)); //"N_states"; //"input.txt";
	strcpy(filename,q1);
    	strcat(filename,q2);
	//filename = concat("N_states", (char)N);
	//filename = "N_states"+"_"+ argv[1] ; //"input.txt";
	//filename.append(argv[1]);
   	filePtr_N_states = fopen(filename,"w");
   	for (int i = 0; i < N; i++) {      		
      			//fprintf(filePtr, "%.3g\t%.3g\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      			fprintf(filePtr_N_states, "%d\t%.5g\n", i, host_rand_float[i]);
      			//fprintf(filePtr2, "%.3g\t%.5g\n", i*N_syn+j, h_conn_w[i*N_syn+j]);   	
   	}
   	
   	/*FILE *filePtr_N_states; 
   	filePtr_N_states = fopen("N_states","r");
   	
   	//FILE *filePtr_N_states_VAL; 
   	//filePtr_N_states_VAL = fopen("N_states_VAL","w");
   	for (int i = 0; i < N; i++) {      		
      			//fprintf(filePtr, "%.3g\t%.3g\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);  			
      			char file_contents[100]; 
      			char file_contents2[100];
      			int N; 
      			if(!feof(filePtr_N_states))   fscanf(filePtr_N_states, "%s %s", file_contents, file_contents2);	
      				//if(sscanf(file_contents, "%*d %d %f ", &h_conn_idx[i*N_syn+j], &h_conn_w[i*N_syn+j]) != 1) puts("scanf() failed!");
      			
      			N = atoi(file_contents);
      			host_rand_float[N] = atof(file_contents2);
      			
      			//fprintf(filePtr_N_states_VAL, "%d\t%.5g\n", i, host_rand_float[i]);
      			//fprintf(filePtr2, "%.3g\t%.5g\n", i*N_syn+j, h_conn_w[i*N_syn+j]);   	
   	}
	printf("Initializing random connectivity matrix values\n");
	cudaGetErrorString(cudaMemcpy(dev_rand_float, host_rand_float, N*sizeof(float), cudaMemcpyHostToDevice));*/
	fclose(filePtr_N_states);
	FILE *filePtr;
 	//filename = "NS_synapses"+"_"+ argv[1] + "_"+ argv[2]  ; //"input.txt";
 	const char* q21 = "NS_synapses";
 	const char* q22 = argv[1];
 	const char* q23 = "_";
 	const char* q24 = argv[2];
 	char* filename2 = (char*) malloc((strlen(q21)+ strlen(q22)+strlen(q23)+strlen(q24))*sizeof(char));
 	strcpy(filename2,q21);
    	strcat(filename2,q22);
    	strcat(filename2,q23);
    	strcat(filename2,q24);
   	filePtr = fopen(filename2,"w");
 
	int idx;
	size_t postsynidx_size;
	int postsynidx[N];						// postsynaptic neuron index
	float rand_float;

	srand((unsigned) time(NULL));
	for(int i=0; i < N; i++){				// run over neurons
		postsynidx_size = N;
		for ( int j=0; j<N; j++ ){			// initialize postsynaptic idx
			postsynidx[j] = j;
		}

		for(int j=0; j<N_syn; j++){			//run over synapses
			idx = (int) rand() % postsynidx_size;
			if ( i < N_exc ){
				rand_float = (1000.0f/N_syn)* 0.5f*(float) rand()/RAND_MAX;
			}else {
				rand_float = (1000.0f/N_syn)* -1.0f*(float) rand()/RAND_MAX;
			}
			h_conn_w[i*N_syn+j] = rand_float;
			h_conn_idx[i*N_syn+j] = postsynidx[idx];
			
			fprintf(filePtr, "%d\t%d\t%d\t%.5g\n", i*N_syn+j, i,  h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
			
			memmove(postsynidx+idx, postsynidx+idx+1, (postsynidx_size-idx-1)*sizeof(int));
			postsynidx_size--;
		}
		if (i%1000==0)
			cout << "neuron " << i << " had connections configured.\n";
	}
	
	fclose(filePtr);
   
   	/*for (int i = 0; i < N; i++) {
      		//y[i] = sqrt(x[i]);
      		for(int j=0; j<N_syn; j++){
      			//fprintf(filePtr, "%.3g\t%.3g\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      			fprintf(filePtr, "%d\t%d\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      			//fprintf(filePtr2, "%.3g\t%.5g\n", i*N_syn+j, h_conn_w[i*N_syn+j]);
   	}
   	}*/

	
	/*FILE * fptr = fopen(filename,"r");
	
  	//unsigned int     string1 ;
  	if(fptr == NULL){
    		printf("Error!! Cannot open file \n" );
    		return 1;}
  	else
    	{
      		printf("File opened successfully :) \n");
      		char file_contents[100]; 
      		char file_contents2[100];     		
      		for (int i = 0; i < N; i++) {      			
      			for(int j=0; j<N_syn; j++){      				
      				//if(!feof(fptr)) fscanf(fptr, "%d%[^\n]" %d %f\n", &string1, &h_conn_idx[i*N_syn+j], &h_conn_w[i*N_syn+j]);		
      				//"%d\t%d\t%.5g\n"	  // "%[^\t]\t%[^\t]\t%[^\n]\n",
      				//if(!feof(fptr))   fscanf(fptr, "%[^\n] ", file_contents);		//"%*s %*s %s ",
      				if(!feof(fptr))   fscanf(fptr, "%*s %s %s ", file_contents, file_contents2);	
      				//if(sscanf(file_contents, "%*d %d %f ", &h_conn_idx[i*N_syn+j], &h_conn_w[i*N_syn+j]) != 1) puts("scanf() failed!");
      				h_conn_idx[i*N_syn+j] = atoi(file_contents);
      				h_conn_w[i*N_syn+j] = atof(file_contents2);
      				
      				//fprintf(filePtr2, "%d\t%d\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      				//cout << file_contents << "\t"  << file_contents2 << "\t" <<  h_conn_idx[i*N_syn+j] << "\t"  << h_conn_w[i*N_syn+j] << "\n";      				 
   			}
   		}  
    	}*/
    	
    	
 
   	/*
   	FILE *filePtr2; 
   	filePtr2 = fopen("floatArray2","w");
   	for (int i = 0; i < N; i++) {
      		//y[i] = sqrt(x[i]);
      		for(int j=0; j<N_syn; j++){
      			//fprintf(filePtr, "%.3g\t%.3g\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      			fprintf(filePtr2, "%d\t%d\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      			//fprintf(filePtr2, "%.3g\t%.5g\n", i*N_syn+j, h_conn_w[i*N_syn+j]);
   	}
   	}*/


	// Copy connectivity matrix to GPU memory
	printf("Retrieving initial connectivity matrix\n");
	cudaGetErrorString(cudaMemcpy(d_conn_w, h_conn_w, N_syn*N*sizeof(float), cudaMemcpyHostToDevice));
	cudaGetErrorString(cudaMemcpy(d_conn_idx, h_conn_idx, N_syn*N*sizeof(int), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	cudaDeviceReset();
	//return 0;	
}

