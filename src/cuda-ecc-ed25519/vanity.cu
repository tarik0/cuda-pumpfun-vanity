#include <vector>
#include <random>
#include <chrono>

#include <iostream>
#include <ctime>

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

/* -- Types ----------------------------------------------------------------- */

typedef struct {
	// CUDA Random States.
	curandState*    states[8];
} config;

/* -- Prototypes, Because C++ ----------------------------------------------- */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_init(unsigned long long int* seed, curandState* state);
void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* execution_count);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
	ed25519_set_verbose(true);

	config vanity;
	vanity_setup(vanity);
	vanity_run(vanity);
}

// SMITH
std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

// SMITH - safe? who knows
unsigned long long int makeSeed() {
    unsigned long long int seed = 0;
    char *pseed = (char *)&seed;

    std::random_device rd;

    for(unsigned int b=0; b<sizeof(seed); b++) {
      auto r = rd();
      char *entropy = (char *)&r;
      pseed[b] = entropy[0];
    }

    return seed;
}

/* -- Vanity Step Functions ------------------------------------------------- */

void vanity_setup(config &vanity) {
	printf("GPU: Initializing Memory\n");
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	// Create random states so kernels have access to random generators
	// while running in the GPU.
	for (int i = 0; i < gpuCount; ++i) {
		cudaSetDevice(i);

		// Fetch Device Properties
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, i);

		// Calculate Occupancy
		int blockSize       = 0,
		    minGridSize     = 0,
		    maxActiveBlocks = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

		// Output Device Details
		// 
		// Our kernels currently don't take advantage of data locality
		// or how warp execution works, so each thread can be thought
		// of as a totally independent thread of execution (bad). On
		// the bright side, this means we can really easily calculate
		// maximum occupancy for a GPU because we don't have to care
		// about building blocks well. Essentially we're trading away
		// GPU SIMD ability for standard parallelism, which CPUs are
		// better at and GPUs suck at.
		//
		// Next Weekend Project: ^ Fix this.
		printf("GPU: %d (%s <%d, %d, %d>) -- W: %d, P: %d, TPB: %d, MTD: (%dx, %dy, %dz), MGS: (%dx, %dy, %dz)\n",
			i,
			device.name,
			blockSize,
			minGridSize,
			maxActiveBlocks,
			device.warpSize,
			device.multiProcessorCount,
		       	device.maxThreadsPerBlock,
			device.maxThreadsDim[0],
			device.maxThreadsDim[1],
			device.maxThreadsDim[2],
			device.maxGridSize[0],
			device.maxGridSize[1],
			device.maxGridSize[2]
		);

                // the random number seed is uniquely generated each time the program 
                // is run, from the operating system entropy

		unsigned long long int rseed = makeSeed();
		printf("Initialising from entropy: %llu\n",rseed);

		unsigned long long int* dev_rseed;
	        cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int));		
                cudaMemcpy( dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice ); 

		cudaMalloc((void **)&(vanity.states[i]), maxActiveBlocks * blockSize * sizeof(curandState));
		vanity_init<<<maxActiveBlocks, blockSize>>>(dev_rseed, vanity.states[i]);
	}

	printf("END: Initializing Memory\n");
}

void vanity_run(config &vanity) {
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	unsigned long long int  executions_total = 0; 
	unsigned long long int  executions_this_iteration; 
	int  executions_this_gpu; 
        int* dev_executions_this_gpu[100];

        int  keys_found_total = 0;
        int  keys_found_this_iteration;
        int* dev_keys_found[100]; // not more than 100 GPUs ok!

	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		auto start  = std::chrono::high_resolution_clock::now();

                executions_this_iteration=0;

		// Run on all GPUs
		for (int g = 0; g < gpuCount; ++g) {
			cudaSetDevice(g);
			// Calculate Occupancy
			int blockSize       = 0,
			    minGridSize     = 0,
			    maxActiveBlocks = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

			int* dev_g;
	                cudaMalloc((void**)&dev_g, sizeof(int));
                	cudaMemcpy( dev_g, &g, sizeof(int), cudaMemcpyHostToDevice ); 

	                cudaMalloc((void**)&dev_keys_found[g], sizeof(int));		
	                cudaMalloc((void**)&dev_executions_this_gpu[g], sizeof(int));		

			vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states[g], dev_keys_found[g], dev_g, dev_executions_this_gpu[g]);

		}

		// Synchronize while we wait for kernels to complete. I do not
		// actually know if this will sync against all GPUs, it might
		// just sync with the last `i`, but they should all complete
		// roughly at the same time and worst case it will just stack
		// up kernels in the queue to run.
		cudaDeviceSynchronize();
		auto finish = std::chrono::high_resolution_clock::now();

		for (int g = 0; g < gpuCount; ++g) {
                	cudaMemcpy( &keys_found_this_iteration, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost ); 
                	keys_found_total += keys_found_this_iteration; 
			//printf("GPU %d found %d keys\n",g,keys_found_this_iteration);

                	cudaMemcpy( &executions_this_gpu, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost ); 
                	executions_this_iteration += executions_this_gpu * ATTEMPTS_PER_EXECUTION; 
                	executions_total += executions_this_gpu * ATTEMPTS_PER_EXECUTION; 
                        //printf("GPU %d executions: %d\n",g,executions_this_gpu);
		}

		// Print out performance Summary
		std::chrono::duration<double> elapsed = finish - start;
		printf("%s Iteration %d Attempts: %llu in %f at %fcps - Total Attempts %llu - keys found %d\n",
			getTimeStr().c_str(),
			i+1,
			executions_this_iteration, //(8 * 8 * 256 * 100000),
			elapsed.count(),
			executions_this_iteration / elapsed.count(),
			executions_total,
			keys_found_total
		);

                if ( keys_found_total >= STOP_AFTER_KEYS_FOUND ) {
                	printf("Enough keys found, Done! \n");
		        exit(0);	
		}	
	}

	printf("Iterations complete, Done!\n");
}

/* -- CUDA Vanity Functions ------------------------------------------------- */

void __global__ vanity_init(unsigned long long int* rseed, curandState* state) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);  
	curand_init(*rseed + id, id, 0, &state[id]);
}

void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* exec_count) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);

        atomicAdd(exec_count, 1);

	// Calculate suffix length
	int suffix_length = 0;
	for(; suffix[suffix_length] != 0; suffix_length++);

	// Local Kernel State
	// ge_p3 A; // Not needed, keypair creation handles it
	curandState localState     = state[id];
	unsigned char seed[32]     = {0};
	unsigned char publick[32]  = {0};
	unsigned char privatek[64] = {0}; // Holds the 64-byte extended private key
	char b58_encoded_key[64]   = {0}; // Buffer for Base58 encoded public key
	size_t b58_pub_size = 64;
	char b58_encoded_private_key[128] = {0}; // Buffer for Base58 encoded private key
	size_t b58_priv_size = 128;

	// Start from an Initial Random Seed
	// NOTE: Insecure random number generator, do not use keys generated by this program in live.
	for (int i = 0; i < 32; i++) {
		seed[i] = (unsigned char)(curand(&localState) * 256);
	}

	for (int i = 0; i < ATTEMPTS_PER_EXECUTION; i++) {
		// Derive the public key and the 64-byte private key from the seed
		ed25519_create_keypair(publick, privatek, seed);

		// Encode public key to Base58
		b58_pub_size = 64; // Reset size
		bool enc_pub_ok = b58enc(b58_encoded_key, &b58_pub_size, publick, 32);

		int match = 0;
		if (enc_pub_ok && b58_pub_size >= suffix_length) {
			match = 1;
			// Compare the end of the Base58 string with the suffix
			for (int j = 0; j < suffix_length; j++) {
				if (b58_encoded_key[b58_pub_size - suffix_length + j] != suffix[j]) {
					match = 0;
					break;
				}
			}
		}

		if (match) {
			atomicAdd(keys_found, 1);
			printf("MATCH SUFFIX GPU %d\n", *gpu);
			// Print Base58 Public Key
			printf("Public (Base58): [");
			for(int k=0; k<b58_pub_size; k++) printf("%c", b58_encoded_key[k]);
			printf("]\n");

			// Encode and Print Private Key in Base58
			b58_priv_size = 128; // Reset size
			bool enc_priv_ok = b58enc(b58_encoded_private_key, &b58_priv_size, privatek, 64);
			if (enc_priv_ok) {
				printf("Secret (Base58): [");
				for (int k = 0; k < b58_priv_size; k++) printf("%c", b58_encoded_private_key[k]);
				printf("]\n");
			} else {
				printf("Secret (Base58): [ENCODING FAILED]\n");
			}
		}

		// Increment the Seed (Super unsafe for real ED25519 keys)
		// Just treat seed as a 256-bit integer and add 1
		for (int j = 0; j < 32; j++) {
			if (seed[j] == 0xFF) {
				seed[j] = 0; // Carry over
			} else {
				seed[j]++;
				break; // No more carry
			}
		}
	}

	// Save the last state.
	state[id] = localState;
}

bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz)
{
	const int8_t b58digits_map[] = {
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1, 0, 1, 2, 3, 4, 5, 6, 7, 8,-1,-1,-1,-1,-1,-1,
		-1, 9,10,11,12,13,14,15,16,-1,17,18,19,20,21,-1,
		22,23,24,25,26,27,28,29,30,31,32,-1,-1,-1,-1,-1,
		-1,33,34,35,36,37,38,39,40,41,42,43,-1,44,45,46,
		47,48,49,50,51,52,53,54,55,56,57,-1,-1,-1,-1,-1,
	};
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

	uint8_t* bin = data;
	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;

	while (zcount < binsz && !bin[zcount])
		++zcount;

	size = (binsz - zcount) * 138 / 100 + 1;
	// Increased buffer size to 128 to handle 64-byte inputs safely
	uint8_t buf[128];
	// Update size check for the larger buffer
	if (size > 128) {
		// Should not happen for 32 or 64 byte inputs, but keep check just in case
		*b58sz = 0; // Indicate error or insufficient buffer
		return false;
	}

	memset(buf, 0, size);

	for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
	{
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
		{
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j) {
				// Otherwise j wraps to maxint which is > high
				break;
			}
		}
	}

	for (j = 0; j < size && !buf[j]; ++j);

	if (*b58sz <= zcount + size - j)
	{
		// Output buffer too small
		*b58sz = zcount + size - j + 1;
		return false;
	}

	if (zcount)
		memset(b58, '1', zcount);
	for (i = zcount; j < size; ++i, ++j)
		b58[i] = b58digits_ordered[buf[j]];
	b58[i] = '\0';
	*b58sz = i;

	return true;
}
