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

// Propagate spikes with dynamic parallelization: AP-algorithm
__global__ void propagatespikes(int spiked, float *d_conn_w, int *d_conn_idx, float *d_Isyn, const int N, const int N_syn){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int dst_idx;
	//AP-Algorithm
	if ( id < N_syn and spiked < N ){
		dst_idx = (int)spiked*N_syn+id;
		atomicAdd(&d_Isyn[d_conn_idx[dst_idx]], d_conn_w[dst_idx]);
	}
}

// State update of neural variables
__global__ void stateupdate(Neuron *neuron,			// Neural parameters of individual neurons
							bool *spike,			// List of booleans to keep spiking neuron indices
							float *d_conn_w,		// Connectivity strengths
							int *d_conn_idx,		// Connection target neuron ids
							float *d_Isyn,			// Synaptic inputs
							const int N,//			// Number of neurons
							const int N_exc,		// Number of excitatory neurons
							//const int N_inh,		// Number of inhibitory neurons
							const int N_syn,		// Number of synapses per neuron
							int gridSize,			// Grid size and block size for the child kernels.
							int blockSize,
							int dynamic,
							int mode){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	//if (id==2499) printf(" id = %d %d %d %d", id, threadIdx.x, blockIdx.x, blockDim.x);
	//printf(" id = %d", threadIdx.x);
	
      if ( id < N ){
	if (!isfinite(neuron[id].I))
		neuron[id].I = 0.f;

		// Selecting the conductances for varying firing regimes
		float ge, gi;
		switch ( mode ){
		case 0: // quiet regime
			ge = 2.5f; gi = 1.0f;
			break;
		case 1: // balanced regime
			ge = 5.0f; gi = 2.0f;
			break;
		case 2: // irregular regime
			ge = 7.5f; gi = 3.0f;
			//ge = 15.0f; gi = 6.0f;
			//ge = 30.0f; gi = 12.0f;
			break;
		}

		if ( id < N_exc ){
			neuron[id].I = ge; //*rand_float[0];		//5.0 for balanced regime, 7.5 for irregular, 2.5 for quiet
		}
		//else if (id >= N_exc and id < N )
		else {
			neuron[id].I = gi;//*rand_float[0];		//2.0 for balanced regime, 3.0 for irregular, 1.0 for quiet
		}

		// Current each neuron receives at a timestep
		// sum of the stochastic and synaptic inputs
		neuron[id].I += d_Isyn[id];

		// update state variables
		neuron[id].v += 0.5f*(0.04f*neuron[id].v*neuron[id].v + 5.0f*neuron[id].v + 140.f - neuron[id].u + neuron[id].I);
		neuron[id].v += 0.5f*(0.04f*neuron[id].v*neuron[id].v + 5.0f*neuron[id].v + 140.f - neuron[id].u + neuron[id].I);
		//neuron[id].v += 1.0f*(0.04f*neuron[id].v*neuron[id].v + 5.0f*neuron[id].v + 140.f - neuron[id].u + neuron[id].I);
		
		neuron[id].u += neuron[id].a*(neuron[id].b*neuron[id].v-neuron[id].u);

		// initialize currents for the next step already
		d_Isyn[id] = 0.f;

		// check if any neuron's membrane potential passed the spiking threshold
		//printf(" v value = %f ", neuron[id].v);
		if ( neuron[id].v >= 30.f or !isfinite(neuron[id].v) ){
			//printf(" hello ");
			spike[id] = true;
			neuron[id].v = neuron[id].c;
			neuron[id].u += neuron[id].d;
			neuron[id].nospks ++;
			// AP-algorithm for spike propagation
			// dynamic: 0 AP-algorithm
			//if ( dynamic == 0  )  //or dynamic == 1
			if ( dynamic == 0 or dynamic == 1 )  //
				propagatespikes<<<gridSize, blockSize>>>(id, d_conn_w, d_conn_idx, d_Isyn, N, N_syn);

		}
		else{
			spike[id] = false;
		}
		
	}	
}

// dynamic: 1  N-algorithm
__global__ void deliverspks1(Neuron *neuron,		// Neural parameters of individual neurons
							bool *spike,			// List of booleans to keep spiking neuron indices
							float *d_conn_w,		// Connectivity strengths
							int *d_conn_idx,		// Connection target neuron ids
							float *d_Isyn,			// Synaptic inputs
							const int N,			// Number of neurons
							const int N_exc,		// Number of excitatory neurons
							const int N_inh,		// Number of inhibitory neurons
							const int N_syn){		// Number of synapses per neuron
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	// N-algorithm
	// a thread for each neuron (presynaptic)
	while (id < N){
		// for each presynaptic neuron
		for (int dst_idx=0; dst_idx<N_syn; dst_idx++){
			// check if there is a spike from presynaptic neurons
			//d_conn_idx[id*N_syn+dst_idx];
			if (spike[id] == true){
				atomicAdd(&d_Isyn[d_conn_idx[id*N_syn+dst_idx]], d_conn_w[id*N_syn+dst_idx]);
			}
		}
		id = id + blockDim.x*gridDim.x;
	}
}

// dynamic: 2 S-algorithm
__global__ void deliverspks2(Neuron *neuron,		// Neural parameters of individual neurons
							bool *spike,			// List of booleans to keep spiking neuron indices
							float *d_conn_w,		// Connectivity strengths
							int *d_conn_idx,		// Connection target neuron ids
							float *d_Isyn,			// Synaptic inputs
							const int N,			// Number of neurons
							const int N_exc,		// Number of excitatory neurons
							const int N_inh,		// Number of inhibitory neurons
							const int N_syn){		// Number of synapses per neuron
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	//int src;
	// S-algorithm
	// a thread for each synapse
	while (idx < N){
		for (int dst_idx=idy; dst_idx<N_syn; dst_idx=dst_idx+blockDim.y*gridDim.y){
			// check if there is a spike from presynaptic neurons
			//d_conn_idx[id*N_syn+dst_idx];
			if (spike[idx] == true){
				atomicAdd(&d_Isyn[d_conn_idx[idx*N_syn+dst_idx]], d_conn_w[idx*N_syn+dst_idx]);
			}
		}
		idx=idx+blockDim.x*gridDim.x;
	}
	
}

////////////dynamic ==3   AB-algo stateupdate
__global__ void AB_stateupdate(Neuron *neuron,			// Neural parameters of individual neurons
							bool *spike,			// List of booleans to keep spiking neuron indices
							float *d_Isyn,			// Synaptic inputs
							const int N,//			// Number of neurons
							const int N_exc,		// Number of excitatory neurons
							int mode){
     int id = threadIdx.x + blockIdx.x * blockDim.x;
	
     //grid_group grid = this_grid(); 
     
     while(id < N){
	  //id = threadIdx.x + block_id * blockDim.x;
	  //id = id + block_id;
	 // if ( id < N )
	 {
	  if (!isfinite(neuron[id].I))
		neuron[id].I = 0.f;

		// Selecting the conductances for varying firing regimes
		float ge, gi;
		switch ( mode ){
		case 0: // quiet regime
			ge = 2.5f; gi = 1.0f;
			break;
		case 1: // balanced regime
			ge = 5.0f; gi = 2.0f;
			break;
		case 2: // irregular regime
			ge = 7.5f; gi = 3.0f;
			//ge = 15.0f; gi = 6.0f;
			//ge = 30.0f; gi = 12.0f;
			break;
		}

		if ( id < N_exc ){
			neuron[id].I = ge;//*rand_float[0];		//5.0 for balanced regime, 7.5 for irregular, 2.5 for quiet
		}
		//else if (id >= N_exc and id < N ){
		else{
			neuron[id].I = gi;//*rand_float[0];		//2.0 for balanced regime, 3.0 for irregular, 1.0 for quiet
		}

		// Current each neuron receives at a timestep
		// sum of the stochastic and synaptic inputs
		neuron[id].I += d_Isyn[id];
			
		// update state variables
		neuron[id].v += 0.5f*(0.04f*neuron[id].v*neuron[id].v + 5.0f*neuron[id].v + 140.f - neuron[id].u + neuron[id].I);
		neuron[id].v += 0.5f*(0.04f*neuron[id].v*neuron[id].v + 5.0f*neuron[id].v + 140.f - neuron[id].u + neuron[id].I);
		neuron[id].u += neuron[id].a*(neuron[id].b*neuron[id].v-neuron[id].u);

		// initialize currents for the next step already
		d_Isyn[id] = 0.f;
		//printf("d_count =  \t" );
		// check if any neuron's membrane potential passed the spiking threshold
		//printf(" v value = %f ", neuron[id].v);
		if ( neuron[id].v >= 30.f or !isfinite(neuron[id].v) ){
			//printf(" hello ");
			spike[id] = true;
			neuron[id].v = neuron[id].c;
			neuron[id].u += neuron[id].d;
			neuron[id].nospks ++;
			//printf("hello\t ");
		}
		else{
			spike[id] = false;   
		}
		
	  }
		id  = id + gridDim.x*blockDim.x;
	  
	 // block_id = block_id + gridDim.x;	
	}//while(id < N);	
}


// //dynamic ==3    AB-algorithm
__global__ void AB_deliverspks(Neuron *neuron,		// Neural parameters of individual neurons
							bool *spike,			// List of booleans to keep spiking neuron indices
							float *d_conn_w,		// Connectivity strengths
							int *d_conn_idx,		// Connection target neuron ids
							float *d_Isyn,			// Synaptic inputs
							const int N,			// Number of neurons
							const int N_exc,		// Number of excitatory neurons
							const int N_inh,		// Number of inhibitory neurons
							const int N_syn		// Number of synapses per neuron
							)
							{
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	//int src;
	// S-algorithm
	// a thread for each synapse
	while (idx < N){
		//int src = (int) (id/N_syn);
		// check if there is a spike from presynaptic neurons
		//if (spike[((id-id%N)/N_syn)] == true){
		//		atomicAdd(&d_Isyn[d_conn_idx[id]], d_conn_w[id]);
		//}
		for (int dst_idx=idy; dst_idx<N_syn; dst_idx=dst_idx+blockDim.y*gridDim.y){
			// check if there is a spike from presynaptic neurons
			//d_conn_idx[id*N_syn+dst_idx];
			if (spike[idx] == true){
				atomicAdd(&d_Isyn[d_conn_idx[idx*N_syn+dst_idx]], d_conn_w[idx*N_syn+dst_idx]);
			}
		}
		idx=idx+blockDim.x*gridDim.x;
	}
}


/////dynamic ==4    under SKL-algorithm  SKL_deliverspks
// State update of neural variables
__device__ void SKL_stateupdate1(Neuron *neuron,			// Neural parameters of individual neurons
							bool *spike,			// List of booleans to keep spiking neuron indices
							float *d_Isyn,			// Synaptic inputs
							const int N,//			// Number of neurons
							const int N_exc,		// Number of excitatory neurons
							int mode){    //, int gridSizeN   
    volatile unsigned int  idx, idy, id;    
    
    idx = threadIdx.x + blockIdx.x * blockDim.x;  idy = threadIdx.y + blockIdx.y * blockDim.y;	
    id = idx*gridDim.y*blockDim.y+idy;
     while(id < N){
	  if (!isfinite(neuron[id].I))
		neuron[id].I = 0.f;
	
		// Selecting the conductances for varying firing regimes
		float ge, gi;
		switch ( mode ){
		case 0: // quiet regime
			ge = 2.5f; gi = 1.0f;
			break;
		case 1: // balanced regime
			ge = 5.0f; gi = 2.0f;
			break;
		case 2: // irregular regime
			ge = 7.5f; gi = 3.0f;
			//ge = 15.0f; gi = 6.0f;
			//ge = 30.0f; gi = 12.0f;
			break;
		}

		if ( id < N_exc ){
			neuron[id].I = ge;//*rand_float[0];		//5.0 for balanced regime, 7.5 for irregular, 2.5 for quiet
		}
		//else if (id >= N_exc and id < N ){
		else {
			neuron[id].I = gi;//*rand_float[0];		//2.0 for balanced regime, 3.0 for irregular, 1.0 for quiet
		}

		// Current each neuron receives at a timestep
		// sum of the stochastic and synaptic inputs
		neuron[id].I += d_Isyn[id];
			
		// update state variables
		neuron[id].v += 0.5f*(0.04f*neuron[id].v*neuron[id].v + 5.0f*neuron[id].v + 140.f - neuron[id].u + neuron[id].I);
		neuron[id].v += 0.5f*(0.04f*neuron[id].v*neuron[id].v + 5.0f*neuron[id].v + 140.f - neuron[id].u + neuron[id].I);
		neuron[id].u += neuron[id].a*(neuron[id].b*neuron[id].v-neuron[id].u);

		// initialize currents for the next step already
		d_Isyn[id] = 0.f;
		//printf("d_count =  \t" );
		// check if any neuron's membrane potential passed the spiking threshold
		//printf(" v value = %f ", neuron[id].v);
		if ( neuron[id].v >= 30.f or !isfinite(neuron[id].v) ){
			//printf(" hello ");
			spike[id] = true;
			neuron[id].v = neuron[id].c;
			neuron[id].u += neuron[id].d;
			neuron[id].nospks ++;
			//printf("hello\t ");
		}
		else{
			spike[id] = false;   
		}
		idx+=blockDim.x*gridDim.x;  idy+=blockDim.y*gridDim.y;
		id = idx*gridDim.y*blockDim.y+idy;
	}		
   
}

__device__ void  spike_count1(bool *d_spikes, int *d_spkcount, const int N){

	//int i;
     	//int block_id = blockIdx.x;
     	//i = threadIdx.x + block_id * blockDim.x;
     	volatile unsigned int  idx, idy, id;    
    
    	idx = threadIdx.x + blockIdx.x * blockDim.x;  idy = threadIdx.y + blockIdx.y * blockDim.y;	
    	id = idx*gridDim.y*blockDim.y+idy;
     	//*d_spkcount = 0;
     	while(id<N){
        	//i = threadIdx.x + block_id * blockDim.x;
		if ( d_spikes[id] == true ){
			atomicAdd((int *)&d_spkcount[0], 1);						
		}		
		//block_id = block_id + gridDim.x;
		//i += blockDim.x*gridDim.x * blockDim.y*gridDim.y;
		//i += blockDim.x*gridDim.x ; //* blockDim.y*gridDim.y;
		idx+=blockDim.x*gridDim.x;  idy+=blockDim.y*gridDim.y;
		id = idx*gridDim.y*blockDim.y+idy;
	}
}

__device__ int d_count;

// //dynamic ==4    SKL-algorithm
__global__ void SKL_deliverspks1(Neuron *neuron,		// Neural parameters of individual neurons
							bool *spike,			// List of booleans to keep spiking neuron indices
							float *d_conn_w,		// Connectivity strengths
							int *d_conn_idx,		// Connection target neuron ids
							float *d_Isyn,			// Synaptic inputs
							const int N,			// Number of neurons
							const int N_exc,		// Number of excitatory neurons
							const int N_syn,		// Number of synapses per neuron
							//int *d_spkcount, 
							int *d_totalspkcount, 	
							const int mode, int d_count1){
	grid_group grid = this_grid();
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
	//unsigned int id = idx*gridDim.y*blockDim.y+idy;	 
	//int d_count = d_count1;
	  if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0 && blockIdx.y==0 && threadIdx.z==0 && blockIdx.z==0) { 
	  	d_count = d_count1;		  	
	  }
	*d_totalspkcount = 0;  //int counter_value=0 ; 	 //
	 grid.sync();
	 __threadfence(); 
	while(d_count){	
	  //////////////////// neuron update  //////////////////////////////////////////////
	  SKL_stateupdate1(neuron, spike, d_Isyn, N, N_exc, mode);
	  grid.sync();
	  __threadfence();
	  //////////////////// spike propagation /////////////////////////////////////////////	  
		while (idx < N){
		//idy =0;
		idy = threadIdx.y + blockIdx.y * blockDim.y;
			while (idy < N_syn){
		//for (idx = threadIdx.x + blockIdx.x* blockDim.x; idx<N; idx=idx+blockDim.x*gridDim.x){		
		//	for (idy = threadIdx.y + blockIdx.y * blockDim.y; idy<N_syn; idy=idy+blockDim.y*gridDim.y){
				// check if there is a spike from presynaptic neurons
				//d_conn_idx[id*N_syn+dst_idx];
				if (spike[idx] == true){
					atomicAdd(&d_Isyn[d_conn_idx[idx*N_syn+idy]], d_conn_w[idx*N_syn+idy]);
				}
				idy+=blockDim.y*gridDim.y;
				//idy=idy+1;
			}
			
			idx+=blockDim.x*gridDim.x;
		}	
	   idx = threadIdx.x + blockIdx.x * blockDim.x;
	   idy = threadIdx.y + blockIdx.y * blockDim.y;
	   //id = idx*gridDim.y*blockDim.y+idy;	
	   //id = idx;	
	   //*d_totalspkcount = 0; 
	   grid.sync();
	   __threadfence();
	  //////////////////// spike count //////////////////////////////////////////////////
	  spike_count1(spike, d_totalspkcount, N);	
	   if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0 && blockIdx.y==0 && threadIdx.z==0 && blockIdx.z==0)  		   
	   {
	  	d_count -= 1;	
	   } 
	  grid.sync(); 
	  __threadfence();
		
   } // while counter loop
	
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
	Neuron *d_Neuron, *h_Neuron;
	h_Neuron = (Neuron *)malloc(N*sizeof(Neuron));
	cudaGetErrorString(cudaMalloc(&d_Neuron, N*sizeof(Neuron)));

	
	// for N neurons to keep spikes
	bool *d_spikes, *h_spikes;
	h_spikes = (bool *)malloc(N*sizeof(bool));
	cudaGetErrorString(cudaMalloc(&d_spikes, N*sizeof(bool)));
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
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_rng, &blockSizeN_rng, stateupdate, 0, 0));
				
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
	/*unsigned int seed_cpu = 1643957724 ; //time(NULL);
	cout<< "seed_cpu " <<seed_cpu<<"\n";
	randgeneratorinit<<<dimGrid_rng, dimBlock_rng>>>(seed_cpu, devStates, N);
	cudaDeviceSynchronize();
	
	randgeneratorinit2<<<dimGrid_rng, dimBlock_rng>>>(dev_rand_float, devStates, N);
	cudaDeviceSynchronize();
	
	// Initialize connectivity matrix on the GPU
	//printf("Initializing random connectivity matrix values\n");
	
	cudaGetErrorString(cudaMemcpy(host_rand_float, dev_rand_float, N*sizeof(float), cudaMemcpyDeviceToHost));
	
	FILE *filePtr_N_states; 
   	filePtr_N_states = fopen("N_states","w");
   	for (int i = 0; i < N; i++) {      		
      			//fprintf(filePtr, "%.3g\t%.3g\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      			fprintf(filePtr_N_states, "%d\t%.5g\n", i, host_rand_float[i]);
      			//fprintf(filePtr2, "%.3g\t%.5g\n", i*N_syn+j, h_conn_w[i*N_syn+j]);   	
   	}
   	*/
   	FILE *filePtr_N_states; 
   	//filePtr_N_states = fopen("N_states","r");
   	
   	const char* q1 = "N_states";
    	const char* q2 = argv[1];

    	//char * qq = (char*) malloc((strlen(q1)+ strlen(q2))*sizeof(char));
    	
	
	char* filename =  (char*) malloc((strlen(q1)+ strlen(q2))*sizeof(char)); //"N_states"; //"input.txt";
	strcpy(filename,q1);
	strcat(filename,q2);
	cout << filename << "\t" << q2 << "\n\n\n";
	filePtr_N_states = fopen(filename,"r");
   	
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
	fclose(filePtr_N_states);
	cudaGetErrorString(cudaMemcpy(dev_rand_float, host_rand_float, N*sizeof(float), cudaMemcpyHostToDevice));
	
	/*int idx;
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
			memmove(postsynidx+idx, postsynidx+idx+1, (postsynidx_size-idx-1)*sizeof(int));
			postsynidx_size--;
		}
		if (i%1000==0)
			cout << "neuron " << i << " had connections configured.\n";
	}*/
	
	
   	/*FILE *filePtr;
 
   	filePtr = fopen("floatArray","w");
 
   	for (int i = 0; i < N; i++) {
      		//y[i] = sqrt(x[i]);
      		for(int j=0; j<N_syn; j++){
      			//fprintf(filePtr, "%.3g\t%.3g\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      			fprintf(filePtr, "%d\t%d\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      			//fprintf(filePtr2, "%.3g\t%.5g\n", i*N_syn+j, h_conn_w[i*N_syn+j]);
   	}
   	}*/

	//const char* filename = "NS_synapses" ; //"input.txt";
	
	FILE * fptr; // = fopen(filename,"r");
	const char* q21 = "NS_synapses";
 	const char* q22 = argv[1];
 	const char* q23 = "_";
 	const char* q24 = argv[2];
 	char* filename2 = (char*) malloc((strlen(q21)+ strlen(q22)+strlen(q23)+strlen(q24))*sizeof(char));
 	strcpy(filename2,q21);
    	strcat(filename2,q22);
    	strcat(filename2,q23);
    	strcat(filename2,q24);
   	fptr = fopen(filename2,"w");
	cout << filename2 << "\n";
	cout << filename2 << "\t" << q22 << "\t" << q24 << "\n\n\n";
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
      				if(!feof(fptr))   fscanf(fptr, "%*s %*s %s %s ", file_contents, file_contents2);	
      				//if(sscanf(file_contents, "%*d %d %f ", &h_conn_idx[i*N_syn+j], &h_conn_w[i*N_syn+j]) != 1) puts("scanf() failed!");
      				h_conn_idx[i*N_syn+j] = atoi(file_contents);
      				h_conn_w[i*N_syn+j] = atof(file_contents2);
      				
      				//fprintf(filePtr2, "%d\t%d\t%.5g\n", i*N_syn+j, h_conn_idx[i*N_syn+j], h_conn_w[i*N_syn+j]);
      				//cout << file_contents << "\t"  << file_contents2 << "\t" <<  h_conn_idx[i*N_syn+j] << "\t"  << h_conn_w[i*N_syn+j] << "\n";      				 
   			}
   		}  
    	}
    	
    	fclose(fptr);
 
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

	// CUDA kernel initiation parameters
	// gridSize and blockSize for N operations
	int blockSizeNNsyn;		// The launch configurator returned block size
	int minGridSizeNNsyn;	// The minimum grid size needed to achieve the
							// maximum occupancy for a full device launch
	int gridSizeNNsyn;		// The actual grid size needed, based on input size
	
	/////////////////////////////////////////  S Algo //////////////////////////////////////
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeNNsyn, &blockSizeNNsyn, deliverspks2, 0, 0);

	// Round up according to array size
	gridSizeNNsyn = (N*N_syn + blockSizeNNsyn - 1) / blockSizeNNsyn;
		
	// run this network configuration
	// for three firing states (mode: 0 quiet, 1 balanced, 2 irregular)
	// with three algorithms (dynamic: 0 AP, 1 N, 2 S 3 AB 4 SKL)
	int d_count = atoi(argv[3]);
	for (int mode=0; mode<3; mode++){
		for (int dynamic=0; dynamic<5; dynamic=dynamic+1){

			// Initialize neural parameters for excitatory and inhibitory neurons
			printf("Initializing neuron parameters\n");
			
			int minGridSize_ini, blockSizeN_ini, gridSizeN_ini;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_ini, &blockSizeN_ini, stateupdate, 0, 0));
				
  			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, initNeuron, blockSizeN_ini, sMemSize));
      			gridSizeN_ini = (N + blockSizeN_ini - 1) / blockSizeN_ini;
  				
  			dim3 dimGrid_ini(gridSizeN_ini, 1, 1),
      			dimBlock_ini(blockSizeN_ini, 1, 1); //
      			occupancy = (numBlocksPerSm * blockSizeN_ini / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      			printf("initNeuron numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      				
      			printf("initNeuron blockSizeN_ini = %d , maxActiveBlocks = %d gridSizeN_ini =%d \n", blockSizeN_ini, numSms * numBlocksPerSm, gridSizeN_ini);
			
			//initNeuron<<<gridSizeN, blockSizeN>>>(N_exc, N, d_Neuron, devStates);
			initNeuron<<<dimGrid_ini, dimBlock_ini>>>(N_exc, N, d_Neuron, dev_rand_float);
			cudaDeviceSynchronize();
			//cudaMemcpyToSymbol(rand_float, &rand_float, sizeof(float *));

			// Copy initial values of neural parameters back to CPU
			printf("Retrieving initial parameter values\n");
			cudaGetErrorString(cudaMemcpy(h_Neuron, d_Neuron, N*sizeof(Neuron), cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();

			// Print out the simulation protocol on the screen
			cout << "===================================================== \n";
			cout << "Neurons: " << N << " Synapses: " << N_syn << " Dynamic (Algorithm): " << dynamic << " Mode (State): " << mode << "\n";
			cout << "===================================================== \n";

			// Open file streams to write data
			// Make folder to save simulation results into
			char buffer[100];
			snprintf(buffer, sizeof(buffer), "Results/");
			mkdir(buffer, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			//snprintf(buffer, sizeof(buffer), "Results/N%dNsyn%dRegime%dAlg%d/", N, N_syn, mode, dynamic);
			snprintf(buffer, sizeof(buffer), "Results/N%dNsyn%dRegime%dAlg%d/", N, N_syn, mode, dynamic);
			mkdir(buffer, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

			// Write initial neural values in the file neuroninfo.csv
			string fname;
			printf("Initializing file streams to write neuron info\n");
			//fname = filename(buffer, "neuroninfo.csv");
			fname = string(buffer) +  "neuroninfo.csv";
			
			ofstream neuroninfo(fname.c_str());

			neuroninfo << "a b c d v u I id nospks" <<  "\n";
			for ( int i=0; i<N; i++ ){
				writeNeuronInfo(neuroninfo, h_Neuron[i]);
			}

			// File to write spike times and spiking neuron ids
			//fname = filename(buffer, "spiketimes.csv");
			fname = string(buffer) + "spiketimes.csv";
			
			ofstream spikes(fname.c_str());
			spikes << "SpkTime NeuronID" << "\n";

			// File to write compute time per timestep  
			//fname = filename(buffer, "computetime.csv");
			fname = string(buffer) + "computetime.csv";
			ofstream computetime(fname.c_str());
			computetime << "TimeStep SpksPerStep TimeKernelUpdate TimeSpent" << "\n";

			// initialize clocks for timekeeping
			clock_t start_sim, end_sim, start, middle, end;
			int spkcount;
			double took, took1, elapsed, elapsed1;
			elapsed = 0;
			elapsed1 = 0;

			cout << "\nStarting simulation and timer...\n";
			int totalspkcount = 0;
			
			unsigned int *d_totalspkcount; int *d_spkcount; // int *d_count;
			//h_spikes = (bool *)malloc(N*sizeof(bool));
			cudaGetErrorString(cudaMalloc(&d_totalspkcount, sizeof(unsigned int)));
			cudaGetErrorString(cudaMalloc(&d_spkcount, sizeof(int)));
			//cudaGetErrorString(cudaMalloc(&d_count, sizeof(int)));
			
			printf("blockSizeN = %d , gridSizeN = %d \n", blockSizeN, gridSizeN);
			printf("blockSizeNNsyn = %d , gridSizeNNsyn = %d \n", blockSizeNNsyn, gridSizeNNsyn);
			
			cudaGetErrorString(cudaMemcpy(d_totalspkcount, &totalspkcount,  sizeof(unsigned int), cudaMemcpyHostToDevice));
				
			//cudaGetErrorString(cudaMemcpy(d_count, &spkcount, sizeof(int), cudaMemcpyHostToDevice));
			cudaGetErrorString(cudaMemcpy(d_spkcount, &spkcount, sizeof(int), cudaMemcpyHostToDevice));
			
			//stateupdate parameters for SOTA - AP N S-algo		
			int minGridSize_su, blockSizeN_su, gridSizeN_su;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_su, &blockSizeN_su, stateupdate, 0, 0));
  			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, stateupdate, blockSizeN_su, sMemSize));
      					
      			gridSizeN_su = (N + blockSizeN_su - 1) / blockSizeN_su;
  				
  			dim3 dimGrid_su(gridSizeN_su, 1, 1),
      				dimBlock_su(blockSizeN_su, 1, 1); //
      			occupancy = (numBlocksPerSm * blockSizeN_su / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      			printf("stateupdate numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      				
      			printf("stateupdate blockSizeN_su = %d , maxActiveBlocks = %d gridSizeN_su =%d \n", blockSizeN_su, numSms * numBlocksPerSm, gridSizeN_su);
      				
      			//AP-algo parameters		
      			int minGridSize_ps, blockSizeN_ps, gridSizeN_ps;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_ps, &blockSizeN_ps, propagatespikes, 0, 0));
  			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, propagatespikes, blockSizeN_ps, sMemSize));
      			gridSizeN_ps = (N + blockSizeN_ps - 1) / blockSizeN_ps;
  				
  			dim3 dimGrid_ps(gridSizeN_ps, 1, 1),
      			dimBlock_ps(blockSizeN_ps, 1, 1); //
      			occupancy = (numBlocksPerSm * blockSizeN_ps / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      			printf("0 AP-algo propagatespikes numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      				
      			printf("0 AP-algo propagatespikes blockSizeN_ps = %d , maxActiveBlocks = %d gridSizeN_ps =%d \n", blockSizeN_ps, numSms * numBlocksPerSm, gridSizeN_ps);
      			
      			//N-algo parameters		
      			int minGridSize_N, blockSizeN_N, gridSizeN_N;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_N, &blockSizeN_N, deliverspks1, 0, 0));
  			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, deliverspks1, blockSizeN_N, sMemSize));
      			gridSizeN_N = (N + blockSizeN_N - 1) / blockSizeN_N;
  			  			
  			dim3 dimGrid_N(gridSizeN_N, 1, 1),
      			dimBlock_N(blockSizeN_N, 1, 1); //
      			occupancy = (numBlocksPerSm * blockSizeN_N / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      			printf("1 N-algo deliverspks1 numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      			printf("1 N-algo deliverspks1 blockSizeN_N = %d , maxActiveBlocks = %d gridSizeN_N =%d \n", blockSizeN_N, numSms * numBlocksPerSm, gridSizeN_N);
      			
      			/// S-algo parameters	
      			int minGridSize_N2, blockSizeN_N2, gridSizeN_N2;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_N2, &blockSizeN_N2, deliverspks2, 0, 0));
  			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, deliverspks2, blockSizeN_N2, sMemSize));
      			gridSizeN_N2 = (N*N_syn + blockSizeN_N2 - 1) / blockSizeN_N2;  //gridSizeNNsyn = (N*N_syn + blockSizeNNsyn - 1) / blockSizeNNsyn;
  				
  			dim3 dimGrid_N2(gridSizeN_N2/2, 2, 1),
      				dimBlock_N2(blockSizeN_N2/16, 16, 1); //
      			occupancy = (numBlocksPerSm * blockSizeN_N2 / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      			printf("2 S-algo deliverspks2 numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      			printf("2 S-algo deliverspks2 blockSizeN_N2 = %d , maxActiveBlocks = %d gridSizeN_N2 =%d \n", blockSizeN_N2, numSms * numBlocksPerSm, gridSizeN_N2);
      			
      			///// state update parameters for AB-algo using persistence
      			int minGridSize_per, blockSizeN_per, gridSizeN_per;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_per, &blockSizeN_per, AB_stateupdate, 0, 0));				
  			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, AB_stateupdate, blockSizeN_per, sMemSize));
      			gridSizeN_per = numSms * numBlocksPerSm;
  			dim3 dimGrid_per(gridSizeN_per, 1, 1),
      				dimBlock_per(blockSizeN_per, 1, 1); //
      			occupancy = (numBlocksPerSm * blockSizeN_per / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      			printf("3 AB_stateupdate numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      			printf("3 AB_stateupdate blockSizeN_per = %d , maxActiveBlocks = %d gridSizeN_per =%d \n", blockSizeN_per, numSms * numBlocksPerSm, gridSizeN_per);  
      			void *kernelArgs_per[] = {
     				 (void *)&d_Neuron,  (void *)&d_spikes, (void *)&d_Isyn, (void *)&N, (void *)&N_exc, (void *)&mode,};
      				
      			///// spike propagation parameters for AB-algo using persistence	
      			int minGridSize_N3, blockSizeN_N3, gridSizeN_N3;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_N3, &blockSizeN_N3, AB_deliverspks, 0, 0));
  			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, AB_deliverspks, blockSizeN_N3, sMemSize));
      				//gridSizeN_N2 = (N*N_syn + blockSizeN_N2 - 1) / blockSizeN_N2;  //gridSizeNNsyn = (N*N_syn + blockSizeNNsyn - 1) / blockSizeNNsyn;
  			gridSizeN_N3 = numSms * numBlocksPerSm;
  			dim3 AB_dimGrid(gridSizeN_N3/4, 4, 1),
      				AB_dimBlock(blockSizeN_N3/4, 4, 1); //
      			occupancy = (numBlocksPerSm * blockSizeN_N3 / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      			printf("3 AB AB_deliverspks numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      			printf("3 AB AB_deliverspks blockSizeN_N3 = %d , maxActiveBlocks = %d gridSizeN_N3 =%d \n", blockSizeN_N3, numSms * numBlocksPerSm, gridSizeN_N3);
      			      			
      			void *AB_kernelArgs[] = {
     				 (void *)&d_Neuron,  (void *)&d_spikes, (void *)&d_conn_w, (void *)&d_conn_idx,
      				(void *)&d_Isyn, (void *)&N, (void *)&N_exc, (void *)&N_inh, (void *)&N_syn, };
			
			/// SKL-algo	parameters	 stateupdate included in the kernel
			int minGridSize_de, blockSizeN_de, gridSizeN_de;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize_de, &blockSizeN_de, SKL_deliverspks1, 0, 0));				
  			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      					&numBlocksPerSm, SKL_deliverspks1, blockSizeN_de, sMemSize));
      			gridSizeN_de = numSms * numBlocksPerSm;
  			dim3 SKL_dimGrid(gridSizeN_de/4, 4, 1),
      				SKL_dimBlock(blockSizeN_de/4,4, 1); //
      			occupancy = (numBlocksPerSm * blockSizeN_de / props.warpSize) / 
                    		(float)(props.maxThreadsPerMultiProcessor / 
                            		props.warpSize);
      			printf("4 SKL SKL_deliverspks numSms = %d  numBlocksPerSm = %d Theoretical occupancy: %f \n", numSms , numBlocksPerSm, occupancy);
      			printf("4 SKL SKL_deliverspks blockSizeN_de = %d , maxActiveBlocks = %d gridSizeN_de =%d \n", blockSizeN_de, numSms * numBlocksPerSm, gridSizeN_de);
							
      			void *SKL_kernelArgs[] = {
     				 (void *)&d_Neuron,  (void *)&d_spikes, (void *)&d_conn_w, (void *)&d_conn_idx,
      				(void *)&d_Isyn, (void *)&N, (void *)&N_exc, (void *)&N_syn, (void *)&d_totalspkcount, (void *)&mode, (void *)&d_count, };    			
			/////////////////////	//////////////////////////////////////////////////////////////
			start_sim = clock();
			
			cudaProfilerStart();		
			
			if (dynamic !=4)
			{

			  // Start main simulation loop  ITER_COUNT
			  for (int tstep=0; tstep<d_count; tstep++){

				spkcount = 0;
				//cudaDeviceSynchronize();
				start = clock();
				// State update
				
				if (dynamic !=3 )  
					stateupdate<<<dimGrid_su, dimBlock_su>>>(d_Neuron, d_spikes, d_conn_w, d_conn_idx, d_Isyn, N, N_exc, N_syn, gridSizeN_ps, blockSizeN_ps, dynamic, mode); // for SOTA 		
				else 
					cudaLaunchCooperativeKernel((void *)AB_stateupdate, dimGrid_per, dimBlock_per, kernelArgs_per, sMemSize, NULL);	//for AB-algo		
						//AB_stateupdate<<<dimGrid_per, dimBlock_per>>>(d_Neuron, d_spikes, d_Isyn, devStates, N, N_exc, mode); 
				//middle = clock();
				
				switch ( dynamic ){
				case 0:
					// if dynamic parallelization, stateupdate kernel already calculated  AP-algorithm
					break;
				case 1:
					// parallelization over neurons: N-algorithm
					//cudaDeviceSynchronize();
					deliverspks1<<<dimGrid_N, dimBlock_N>>> (d_Neuron, d_spikes, d_conn_w, d_conn_idx, d_Isyn, N, N_exc, N_inh, N_syn);
					//stateupdate<<<dimGrid_su, dimBlock_su>>>(d_Neuron, d_spikes, d_conn_w, d_conn_idx, d_Isyn, N, N_exc, N_syn, gridSizeN_ps, blockSizeN_ps, dynamic, mode); // for SOTA 
					break;
				case 2:
					// parallelization over synapses: S-algorithm
					//cudaDeviceSynchronize();
					deliverspks2<<<dimGrid_N2, dimBlock_N2>>> (d_Neuron, d_spikes, d_conn_w, d_conn_idx, d_Isyn, N, N_exc, N_inh, N_syn);
					break;
					
				case 3:
					// parallelization over synapses: AB-algorithm     Modified S-algorithm using persistence   
					//AB_deliverspks<<<AB_dimGrid, AB_dimBlock>>> (d_Neuron, d_spikes, d_conn_w, d_conn_idx, d_Isyn, N, N_exc, N_inh, N_syn, gridSizeNNsyn);
					cudaLaunchCooperativeKernel((void *)AB_deliverspks,
                                              AB_dimGrid, AB_dimBlock, AB_kernelArgs,
                                              sMemSize, NULL);				
				}
				//cudaDeviceSynchronize();
				//end = clock();
				middle = clock();
				cudaMemcpy(h_spikes, d_spikes, N*sizeof(bool), cudaMemcpyDeviceToHost);
				took = double(middle-start) / CLOCKS_PER_SEC * 1000;
				
				elapsed += took;
				//start_sim1 = clock();
				middle = clock();
				cudaMemcpy(h_spikes, d_spikes, N*sizeof(bool), cudaMemcpyDeviceToHost);
				//end_sim1 = clock();
				end = clock();
				//cudaDeviceSynchronize();
				for ( int i=0; i<N; i++){
					if ( h_spikes[i] == true ){
						spikes << tstep << " " << i << "\n";
						spkcount++;
					}
				}

				took1 = double(end-middle) / CLOCKS_PER_SEC * 1000;
				elapsed1 += took1;
				

				computetime << tstep << " " << spkcount << " " << took1 << " " << took << "\n";
				//printf("spkcount =%d  totalspkcount =%d\t", spkcount, totalspkcount);
				totalspkcount += spkcount;

			  }
			  end_sim = clock();
			}
			
			else
			{
				start_sim = clock();				
				if(dynamic ==4) {	  //dynamic ==4    SKL-algorithm                               	
                               	totalspkcount = 0;	// *totalspkcount_new = 0; 
                               	elapsed = 0;	
                               	start = clock();                               	
                               	cudaLaunchCooperativeKernel((void *)SKL_deliverspks1,
                                              SKL_dimGrid, SKL_dimBlock, SKL_kernelArgs,
                                              sMemSize, NULL);
                                      end = clock();
                                      //cudaMemcpy(&totalspkcount, d_totalspkcount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                                      middle = clock();
					
				       cudaMemcpy(&totalspkcount, d_totalspkcount, sizeof(unsigned int), cudaMemcpyDeviceToHost); //d_spkcount
			   		//totalspkcount = (int)*totalspkcount_new;	
			   		//end_sim = clock();	
			   		took = double(middle-start) / CLOCKS_PER_SEC * 1000;
					
				     	elapsed += took;
                                 }
                                 	
                                  // start = clock();   
                                   middle = clock();	
                                   cudaMemcpy(&totalspkcount, d_totalspkcount, sizeof(unsigned int), cudaMemcpyDeviceToHost); //d_spkcount
                                   end = clock();
                                   
                                  
                                   
                                   took1 = double(end-middle) / CLOCKS_PER_SEC * 1000;
                                   elapsed1 += took1;
				    end_sim = clock();			//totalspkcount = spkcount;	
			  	//cudaDeviceSynchronize();	
				
				//computetime << tstep << " " << spkcount << " " << took1 << " " << took << "\n";
				//totalspkcount += spkcount;			   
			   
			}
			//end_sim = clock();			
			cudaProfilerStop();
			cout << "\nClocks per sec is " << CLOCKS_PER_SEC << "\n";
			cout << "\nEnd of simulation...\n";
			cout << "Simulation took: " << double(end_sim-start_sim) / CLOCKS_PER_SEC * 1000<< " ms.\n";
			cout << "Without data transfers: " << elapsed << " ms.\n";
			cout << "Only data transfers: " << elapsed1 << " ms.\n";
			cout << "Total number of spikes: " << totalspkcount << "\n\n\n";

			// Write simulation overview in the file sim_overview.csv
			//ofstream sim_overview(filename(buffer, "sim_overview.csv"));
			ofstream sim_overview(string(buffer) + "sim_overview.csv");
			sim_overview << "N elapsed(ms) elapsedwithdata(ms) totalspkcount elapsed_only_data(ms)" <<  "\n";
			sim_overview << N << " " << elapsed << " " << double(end_sim-start_sim) / CLOCKS_PER_SEC * 1000 << " " << totalspkcount << " " << elapsed1;
                       //sim_overview.close();

			// Write neural values in the file neuroninfo.csv
			//ofstream neuroninfoafter(filename(buffer, "neuroninfoafter.csv"));
			ofstream neuroninfoafter(string(buffer) + "neuroninfoafter.csv");
			if ( ! neuroninfoafter.is_open() ){
				//std::cout <<  buffer << std::endl;
    				std::cerr << "Error! Failed to open file "  << std::endl;
    				return 1;
			}
			neuroninfoafter << "a b c d v u I id nospks" <<  "\n";
			for ( int i=0; i<N; i++ ){
				writeNeuronInfo(neuroninfoafter, h_Neuron[i]);
			}
			//neuroninfoafter.close();
		}
	}
	cudaDeviceReset();	
}
