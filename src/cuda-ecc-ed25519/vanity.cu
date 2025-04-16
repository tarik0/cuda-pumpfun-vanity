#include <vector>
#include <random>
#include <chrono>
#include <cstring> // For strlen, strcmp
#include <sstream> // For std::stringstream

#include <iostream>
#include <ctime>
#include <iomanip> // For std::setw

#include <assert.h>
#include <inttypes.h>
// #include <pthread.h> // Not used
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

/* -- Error Checking -------------------------------------------------------- */

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/* -- Types ----------------------------------------------------------------- */

// Structure to hold results found on the GPU
typedef struct {
    unsigned char public_key[32];
    unsigned char private_key[64]; // 64-byte expanded secret key
} FoundKeyPair;


typedef struct {
	int gpu_count;
	// CUDA Random States (one per GPU)
	std::vector<curandState*> states;
	// Execution counters (one per GPU)
	std::vector<int*> dev_executions_this_gpu;
	// Found key counters (one per GPU)
	std::vector<int*> dev_keys_found_index;
	// GPU ID pointers (one per GPU)
	std::vector<int*> dev_g;
	// Found key results buffer (one per GPU)
	std::vector<FoundKeyPair*> dev_found_keys;
} config;

/* -- Prototypes, Because C++ ----------------------------------------------- */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void            vanity_cleanup(config& vanity);
void __global__ vanity_init(unsigned long long int* seed, curandState* state, int N);
void __global__ vanity_scan(curandState* state, int* keys_found_index, int* gpu, int* exec_count, FoundKeyPair* results_buffer, int max_results);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);
bool host_b58enc(char* b58, size_t* b58sz, const unsigned char* data, size_t binsz);

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
	// ed25519_set_verbose(true); // Can be noisy

	config vanity;
	vanity_setup(vanity);
	vanity_run(vanity);
	vanity_cleanup(vanity);
	return 0;
}

// SMITH - Get current timestamp as string
std::string getTimeStr(){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// SMITH - Generate seed from hardware entropy
unsigned long long int makeSeed() {
    unsigned long long int seed = 0;
    std::random_device rd;
    // Fill the seed byte by byte with random data
    for (size_t i = 0; i < sizeof(seed); ++i) {
        ((char*)&seed)[i] = static_cast<char>(rd());
    }
    return seed;
}

/* -- Vanity Step Functions ------------------------------------------------- */

void vanity_setup(config &vanity) {
	printf("GPU: Initializing Memory\n");
	gpuErrchk(cudaGetDeviceCount(&vanity.gpu_count));

	if (vanity.gpu_count == 0) {
		fprintf(stderr, "Error: No CUDA-enabled GPUs found.\n");
		exit(EXIT_FAILURE);
	}
	printf("Found %d CUDA-enabled GPU(s)\n", vanity.gpu_count);

	// Resize vectors based on actual GPU count
	vanity.states.resize(vanity.gpu_count);
	vanity.dev_executions_this_gpu.resize(vanity.gpu_count);
	vanity.dev_keys_found_index.resize(vanity.gpu_count);
	vanity.dev_g.resize(vanity.gpu_count);
	vanity.dev_found_keys.resize(vanity.gpu_count);


	// Create random states and allocate other per-GPU resources
	for (int i = 0; i < vanity.gpu_count; ++i) {
		gpuErrchk(cudaSetDevice(i));

		// Fetch Device Properties
		cudaDeviceProp device;
		gpuErrchk(cudaGetDeviceProperties(&device, i));

		// Calculate Occupancy
		int blockSize       = 0,
		    minGridSize     = 0,
		    maxActiveBlocks = 0;
		// Note: Occupancy calculated based on the *new* vanity_scan signature
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0));
		gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0));

		// Output Device Details
		printf("GPU %d: %s (Compute %d.%d) | MP: %d | Warpsize: %d | Max Threads/Block: %d | Optimal Blocksize: %d | Gridsize: %d | Max Active Blocks: %d\n",
			i,
			device.name,
			device.major, device.minor,
			device.multiProcessorCount,
			device.warpSize,
			device.maxThreadsPerBlock,
			blockSize,
			minGridSize,
			maxActiveBlocks
		);

		// Generate a unique seed for this GPU's initializer
		unsigned long long int rseed = makeSeed();
		printf("GPU %d: Initialising RNG from entropy: %llu\n", i, rseed);

		unsigned long long int* dev_rseed;
		gpuErrchk(cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int)));
		gpuErrchk(cudaMemcpy(dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

		// Allocate CURAND states for this GPU
		size_t num_threads = maxActiveBlocks * blockSize;
		gpuErrchk(cudaMalloc((void **)&(vanity.states[i]), num_threads * sizeof(curandState)));

		// Initialize CURAND states
		vanity_init<<<maxActiveBlocks, blockSize>>>(dev_rseed, vanity.states[i], num_threads);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize()); // Ensure init is done before freeing seed

		gpuErrchk(cudaFree(dev_rseed)); // Free seed memory now

		// Allocate other per-GPU buffers
		gpuErrchk(cudaMalloc((void**)&vanity.dev_g[i], sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&vanity.dev_keys_found_index[i], sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&vanity.dev_executions_this_gpu[i], sizeof(int)));
		// Allocate buffer for found keys (size based on STOP_AFTER_KEYS_FOUND)
		gpuErrchk(cudaMalloc((void**)&vanity.dev_found_keys[i], STOP_AFTER_KEYS_FOUND * sizeof(FoundKeyPair)));
	}

	printf("END: Initializing Memory\n");
}

void print_found_key(const FoundKeyPair& key_pair, int gpu_id) {
    char b58_encoded_pub[64]; // Buffer for Base58 public key (32 bytes -> max ~44 chars)
    size_t b58_pub_size = sizeof(b58_encoded_pub);
    char b58_encoded_priv[128]; // Buffer for Base58 private key (64 bytes -> max ~88 chars)
    size_t b58_priv_size = sizeof(b58_encoded_priv);

    // Host-side Base58 encoding
    bool pub_enc_ok = host_b58enc(b58_encoded_pub, &b58_pub_size, key_pair.public_key, 32);
    bool priv_enc_ok = host_b58enc(b58_encoded_priv, &b58_priv_size, key_pair.private_key, 64);


    printf("\n--- MATCH FOUND (GPU %d) ---\n", gpu_id);

    // Print Public Key (Base58)
    printf("Public (Base58): [");
    if (pub_enc_ok) {
        // Use %.*s to print exactly b58_pub_size characters
        printf("%.*s", (int)b58_pub_size, b58_encoded_pub);
    } else {
        printf("ENCODING FAILED - Needed buffer size: %zu", b58_pub_size);
    }
    printf("]\n");


    // Print Private Key (Base58)
    printf("Secret (Base58): [");
    if (priv_enc_ok) {
        printf("%.*s", (int)b58_priv_size, b58_encoded_priv);
    } else {
        printf("ENCODING FAILED - Needed buffer size: %zu", b58_priv_size);
    }
    printf("]\n");

    printf("--------------------------\n");
}


void vanity_run(config &vanity) {
	unsigned long long int  executions_total = 0;
	int  keys_found_total = 0;

	// Allocate host buffer for results from all GPUs
	std::vector<FoundKeyPair> host_found_keys(vanity.gpu_count * STOP_AFTER_KEYS_FOUND);
	std::vector<int> host_keys_found_index(vanity.gpu_count);

	printf("\nStarting vanity search for suffix '%s'\n", suffix); // Use suffix from config.h

	for (int iter = 0; iter < MAX_ITERATIONS && keys_found_total < STOP_AFTER_KEYS_FOUND; ++iter) {
		auto start  = std::chrono::high_resolution_clock::now();

		unsigned long long int executions_this_iteration = 0;
		int keys_found_this_iteration_total = 0; // Keys found across all GPUs in this iteration


		// Reset device counters and launch kernels on all GPUs
		for (int g = 0; g < vanity.gpu_count; ++g) {
			gpuErrchk(cudaSetDevice(g));

			// Calculate Occupancy (can be done once in setup if properties don't change)
			int blockSize       = 0, minGridSize = 0, maxActiveBlocks = 0;
			gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0));
			gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0));

			// Update device memory for GPU ID
			gpuErrchk(cudaMemcpy(vanity.dev_g[g], &g, sizeof(int), cudaMemcpyHostToDevice));

			// Reset device counters for this iteration
			gpuErrchk(cudaMemset(vanity.dev_keys_found_index[g], 0, sizeof(int)));
			gpuErrchk(cudaMemset(vanity.dev_executions_this_gpu[g], 0, sizeof(int)));
			// No need to memset the results buffer itself

			// Launch kernel
			vanity_scan<<<maxActiveBlocks, blockSize>>>(
				vanity.states[g],
				vanity.dev_keys_found_index[g],
				vanity.dev_g[g],
				vanity.dev_executions_this_gpu[g],
				vanity.dev_found_keys[g],
				STOP_AFTER_KEYS_FOUND // Max results this GPU can store
			);
			gpuErrchk(cudaPeekAtLastError()); // Check for launch errors
		}

		// Synchronize all GPUs after launching all kernels
		gpuErrchk(cudaDeviceSynchronize());
		auto finish = std::chrono::high_resolution_clock::now();

		// Process results from all GPUs
		for (int g = 0; g < vanity.gpu_count; ++g) {
			gpuErrchk(cudaSetDevice(g)); // Set context for DtoH copy

			// Copy back the number of keys found by this GPU in this iteration
			int keys_found_this_gpu = 0;
			gpuErrchk(cudaMemcpy(&keys_found_this_gpu, vanity.dev_keys_found_index[g], sizeof(int), cudaMemcpyDeviceToHost));
			keys_found_this_iteration_total += keys_found_this_gpu;
			host_keys_found_index[g] = keys_found_this_gpu; // Store count for this GPU

			// Copy back execution count for this GPU
			int executions_this_gpu = 0;
			gpuErrchk(cudaMemcpy(&executions_this_gpu, vanity.dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost));
			executions_this_iteration += (unsigned long long)executions_this_gpu * ATTEMPTS_PER_EXECUTION;

			// Copy back the actual found keys if any
			if (keys_found_this_gpu > 0) {
				if (keys_found_this_gpu > STOP_AFTER_KEYS_FOUND) {
					fprintf(stderr, "Warning: GPU %d reported %d keys, but buffer only holds %d. Clamping.\n", g, keys_found_this_gpu, STOP_AFTER_KEYS_FOUND);
					keys_found_this_gpu = STOP_AFTER_KEYS_FOUND;
					host_keys_found_index[g] = keys_found_this_gpu; // Update host count
				}
				// Copy to the correct offset in the host buffer
				gpuErrchk(cudaMemcpy(host_found_keys.data() + g * STOP_AFTER_KEYS_FOUND,
				                     vanity.dev_found_keys[g],
				                     keys_found_this_gpu * sizeof(FoundKeyPair),
				                     cudaMemcpyDeviceToHost));
			}
		}

		executions_total += executions_this_iteration;

		// Print keys found in *this* iteration from the host buffer
		int current_total_keys_before_iteration = keys_found_total;
		for (int g = 0; g < vanity.gpu_count; ++g) {
			for (int k = 0; k < host_keys_found_index[g]; ++k) {
				if (keys_found_total < STOP_AFTER_KEYS_FOUND) {
					print_found_key(host_found_keys[g * STOP_AFTER_KEYS_FOUND + k], g);
					keys_found_total++;
				} else {
					// Avoid printing more than requested if limit reached mid-iteration
					break;
				}
			}
			if (keys_found_total >= STOP_AFTER_KEYS_FOUND) break;
		}


		// Print out performance Summary
		std::chrono::duration<double> elapsed = finish - start;
		double rate = (elapsed.count() > 0) ? (executions_this_iteration / elapsed.count()) : 0.0;
		printf("%s Iter %d | Found: %d (+%d) | Speed: %.2f Mcps | Total Att: %llu | Elapsed: %.3fs\n",
			getTimeStr().c_str(),
			iter + 1,
			keys_found_total,
			keys_found_total - current_total_keys_before_iteration, // Keys found this iteration
			rate / 1.0e6, // Rate in Million calculations per second
			executions_total,
			elapsed.count()
		);

        // Check if we've found enough keys globally
        if (keys_found_total >= STOP_AFTER_KEYS_FOUND) {
            printf("\nTarget key count (%d) reached. Finishing.\n", STOP_AFTER_KEYS_FOUND);
            break; // Exit the iteration loop
        }
	}

	if (keys_found_total < STOP_AFTER_KEYS_FOUND) {
		printf("\nMaximum iterations (%d) reached. Finishing.\n", MAX_ITERATIONS);
	}
}

void vanity_cleanup(config &vanity) {
	printf("\nCleaning up GPU resources...\n");
	for (int g = 0; g < vanity.gpu_count; ++g) {
		gpuErrchk(cudaSetDevice(g));
		// Free all allocated device memory
		gpuErrchk(cudaFree(vanity.states[g]));
		gpuErrchk(cudaFree(vanity.dev_g[g]));
		gpuErrchk(cudaFree(vanity.dev_keys_found_index[g]));
		gpuErrchk(cudaFree(vanity.dev_executions_this_gpu[g]));
		gpuErrchk(cudaFree(vanity.dev_found_keys[g]));
	}
	printf("Cleanup complete.\n");
}


/* -- CUDA Vanity Functions ------------------------------------------------- */

// Initialize CURAND states for N threads
void __global__ vanity_init(unsigned long long int* rseed, curandState* state, int N) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < N) { // Ensure we don't initialize out of bounds
		curand_init(*rseed + id, id, 0, &state[id]);
	}
}

// Main kernel: Generate keys and check for suffix match
void __global__ vanity_scan(curandState* state, int* keys_found_index, int* gpu, int* exec_count, FoundKeyPair* results_buffer, int max_results) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;

    atomicAdd(exec_count, 1); // Count kernel executions (blocks*threads)

	curandState localState = state[id]; // Load state into local thread memory

	// Local Kernel State
	unsigned char seed[32];             // 32-byte seed for key generation
	unsigned char publick[32];          // Public key output
	unsigned char privatek[64];         // Expanded private key output (64 bytes)
	char b58_encoded_key[64];           // Buffer for Base58 encoded public key
	size_t b58_pub_size;                // Size of encoded public key
	// Note: private key encoding removed from kernel, done on host if needed

	for (int i = 0; i < ATTEMPTS_PER_EXECUTION; i++) {
		// 1. Generate a random 32-byte seed for this attempt
		// Use curand_uniform4 for efficiency if possible, otherwise fill byte-by-byte
        // Example using byte-by-byte (simple, maybe not fastest):
        unsigned int* seed_as_uint = (unsigned int*)seed;
        for (int k = 0; k < 8; ++k) { // Generate 8 * 4 = 32 bytes
             seed_as_uint[k] = curand(&localState);
        }
        // Alternative: Use curand_uniform4 if aligned access is guaranteed
        // float4 r = curand_uniform4(&localState); // Example - needs conversion


		// 2. Derive the public key and the 64-byte private key from the random seed
		ed25519_create_keypair(publick, privatek, seed);

		// 3. Encode public key to Base58
		b58_pub_size = 64; // Reset size for b58enc
		bool enc_pub_ok = b58enc(b58_encoded_key, &b58_pub_size, publick, 32);

		// 4. Check for suffix match using precomputed length
		bool match = false;
		if (enc_pub_ok && b58_pub_size >= suffix_length) {
			match = true;
			// Compare the end of the Base58 string with the suffix
			for (int j = 0; j < suffix_length; j++) {
				if (b58_encoded_key[b58_pub_size - suffix_length + j] != suffix[j]) {
					match = false;
					break;
				}
			}
		}

		// 5. If match found, store result in the output buffer
		if (match) {
			// Atomically get the next available index in the results buffer
			int result_idx = atomicAdd(keys_found_index, 1);

			// Ensure we don't write out of bounds
			if (result_idx < max_results) {
				// Copy keys to the results buffer at the obtained index
				for(int k=0; k<32; k++) results_buffer[result_idx].public_key[k] = publick[k];
				for(int k=0; k<64; k++) results_buffer[result_idx].private_key[k] = privatek[k];

				// Original printf replaced by storing result
				// printf("MATCH SUFFIX GPU %d\n", *gpu); // REMOVED
                // Print logic moved to host side after kernel completion
			}
            // If result_idx >= max_results, we simply drop the key to avoid buffer overflow.
            // The host will see keys_found_index > max_results and potentially warn.
		}

		// 6. Seed increment logic REMOVED - we generate a new random seed each iteration
	}

	// Save the final state back to global memory
	state[id] = localState;
}


// Base58 encoding function (unchanged, keep __device__)
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz)
{
	// Base58 character set for encoding
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

	// Use cuda_memset or equivalent if available/necessary for device code,
	// standard memset might work depending on CUDA version/arch but safer to be explicit if needed.
	// For now, assume standard memset works or compiler handles it.
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
		*b58sz = zcount + size - j + 1; // Return required size
		return false;
	}

	if (zcount)
		memset(b58, '1', zcount); // Assuming standard memset works here too
	for (i = zcount; j < size; ++i, ++j)
		b58[i] = b58digits_ordered[buf[j]];
	b58[i] = '\0';
	*b58sz = i;

	return true;
}

// Host-side Base58 encoding function (copied from device version)
bool host_b58enc(char* b58, size_t* b58sz, const unsigned char* data, size_t binsz)
{
    // Base58 character set for encoding
    const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    const uint8_t* bin = data;
    int carry;
    size_t i, j, high, zcount = 0;
    size_t size;

    // Find number of leading zeros
    while (zcount < binsz && !bin[zcount])
        ++zcount;

    // Calculate required buffer size
    size = (binsz - zcount) * 138 / 100 + 1;
    // Use a dynamically sized buffer on host or ensure stack buffer is large enough
    // Using a fixed-size stack buffer known to be large enough for 64-byte input (results in ~88 chars + prefix)
    uint8_t buf[128]; // Sufficient for 64-byte input
    if (size > sizeof(buf)) {
        // This shouldn't happen for inputs up to 64 bytes but good to check
        *b58sz = size; // Return required size
        return false; // Indicate buffer overflow potential
    }

    memset(buf, 0, size); // Initialize buffer to zeros

    // Main conversion loop
    for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
    {
        for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
        {
            carry += 256 * buf[j];
            buf[j] = carry % 58;
            carry /= 58;
            if (!j) {
                // Prevent wrap-around
                break;
            }
        }
    }

    // Skip leading zeros in the result buffer
    for (j = 0; j < size && !buf[j]; ++j);

    // Check if the output buffer (b58) is large enough
    if (*b58sz <= zcount + size - j)
    {
        // Output buffer too small, report required size
        *b58sz = zcount + size - j + 1;
        return false;
    }

    // Add leading '1' characters for input leading zeros
    if (zcount)
        memset(b58, '1', zcount);

    // Copy the result digits to the output buffer
    for (i = zcount; j < size; ++i, ++j)
        b58[i] = b58digits_ordered[buf[j]];

    // Null-terminate the output string
    b58[i] = '\0';
    // Set the final size of the encoded string
    *b58sz = i;

    return true;
}
