#include <vector>
#include <random>
#include <chrono>
#include <cstring> // For strlen, strcmp, memcpy, memset
#include <sstream> // For std::stringstream
#include <string>  // For std::string

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
    // unsigned char private_key[64]; // OLD: 64-byte expanded secret key
    unsigned char seed[32];         // NEW: Original 32-byte seed (this is what wallets import)
} FoundKeyPair;


typedef struct {
	int gpu_count;
	// CUDA Random States (one per GPU)
	std::vector<curandState*> states;
	// Execution counters (one per GPU)
	std::vector<int*> dev_executions_this_gpu;
	// Found key counters (one per GPU)
	std::vector<int*> dev_keys_found_index;
	// Found key results buffer (one per GPU)
	std::vector<FoundKeyPair*> dev_found_keys;

    // Launch parameters (calculated once per GPU)
    std::vector<int> blockSizes;      // Store optimal block size per GPU
    std::vector<int> maxActiveBlocks; // Store max active blocks per GPU

} config;

/* -- Prototypes, Because C++ ----------------------------------------------- */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void            vanity_cleanup(config& vanity);
void __global__ vanity_init(unsigned long long int* seed, curandState* state, int N);
// Updated signature: Removed unused 'gpu' pointer
void __global__ vanity_scan(curandState* state, int* keys_found_index, int* exec_count, FoundKeyPair* results_buffer, int max_results);
// Device-side Base58 encoder prototype
bool __device__ b58enc(char* b58, size_t* b58sz, const uint8_t* data, size_t binsz);
// Host-side Base58 encoder prototype (const correct)
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
        // Cast carefully to avoid sign extension issues if char is signed
        ((unsigned char*)&seed)[i] = static_cast<unsigned char>(rd());
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
	vanity.dev_found_keys.resize(vanity.gpu_count);
    vanity.blockSizes.resize(vanity.gpu_count);
    vanity.maxActiveBlocks.resize(vanity.gpu_count);

	// Create random states and allocate other per-GPU resources
	for (int i = 0; i < vanity.gpu_count; ++i) {
		gpuErrchk(cudaSetDevice(i));

		// Fetch Device Properties
		cudaDeviceProp device;
		gpuErrchk(cudaGetDeviceProperties(&device, i));

		// Calculate Occupancy and store it
		int blockSize       = 0,
		    minGridSize     = 0,
		    maxActiveBlocksPerSM = 0; // Renamed for clarity
		// Note: Occupancy calculated based on the *updated* vanity_scan signature
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)vanity_scan, 0, 0));
        // Calculate max active blocks *per SM*
		gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, (void*)vanity_scan, blockSize, 0));
        // Total max active blocks for the launch grid
        int totalMaxActiveBlocks = maxActiveBlocksPerSM * device.multiProcessorCount;

        // Store calculated values
        vanity.blockSizes[i] = blockSize;
        vanity.maxActiveBlocks[i] = totalMaxActiveBlocks; // Store total blocks for launch

		// Output Device Details
		printf("GPU %d: %s (Compute %d.%d) | MP: %d | Warpsize: %d | Max Threads/Block: %d\n",
			i, device.name, device.major, device.minor,
			device.multiProcessorCount, device.warpSize, device.maxThreadsPerBlock);
        printf("       Optimal Blocksize: %d | Min Grid Size (Blocks): %d | Max Active Blocks/SM: %d | Launch Grid (Blocks): %d\n",
			blockSize, minGridSize, maxActiveBlocksPerSM, totalMaxActiveBlocks);


		// Generate a unique seed for this GPU's initializer
		unsigned long long int rseed = makeSeed();
		printf("GPU %d: Initialising RNG from entropy: %llu\n", i, rseed);

		unsigned long long int* dev_rseed;
		gpuErrchk(cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int)));
		gpuErrchk(cudaMemcpy(dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

		// Allocate CURAND states for this GPU
		// Size based on total threads needed = total blocks * threads per block
		size_t num_threads = (size_t)totalMaxActiveBlocks * blockSize;
		gpuErrchk(cudaMalloc((void **)&(vanity.states[i]), num_threads * sizeof(curandState)));

		// Initialize CURAND states
		vanity_init<<<totalMaxActiveBlocks, blockSize>>>(dev_rseed, vanity.states[i], num_threads);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize()); // Ensure init is done before freeing seed

		gpuErrchk(cudaFree(dev_rseed)); // Free seed memory now

		// Allocate other per-GPU buffers
		gpuErrchk(cudaMalloc((void**)&vanity.dev_keys_found_index[i], sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&vanity.dev_executions_this_gpu[i], sizeof(int)));
		// Allocate buffer for found keys (size based on STOP_AFTER_KEYS_FOUND)
		gpuErrchk(cudaMalloc((void**)&vanity.dev_found_keys[i], STOP_AFTER_KEYS_FOUND * sizeof(FoundKeyPair)));
	}

	printf("END: Initializing Memory\n");
}

// Modified to print the 32-byte seed correctly
void print_found_key(const FoundKeyPair& key_pair, int gpu_id) {
    char b58_encoded_pub[64]; // Buffer for Base58 public key (32 bytes -> max ~44 chars + null)
    size_t b58_pub_size = sizeof(b58_encoded_pub);
    char b58_encoded_seed[64]; // Buffer for Base58 seed (32 bytes -> max ~44 chars + null)
    size_t b58_seed_size = sizeof(b58_encoded_seed);

    // Host-side Base58 encoding
    bool pub_enc_ok = host_b58enc(b58_encoded_pub, &b58_pub_size, key_pair.public_key, 32);
    bool seed_enc_ok = host_b58enc(b58_encoded_seed, &b58_seed_size, key_pair.seed, 32); // Encode the seed


    printf("\n--- MATCH FOUND (GPU %d) ---\n", gpu_id);

    // Print Public Key (Base58)
    printf("Public Key (Base58) : ");
    if (pub_enc_ok) {
        // Use %.*s to print exactly b58_pub_size characters
        printf("[%.*s]\n", (int)b58_pub_size, b58_encoded_pub);
    } else {
        printf("[ENCODING FAILED - Needed buffer size: %zu]\n", b58_pub_size);
    }

    // Print Secret Seed (Base58) - This is what wallets import
    printf("Secret Seed (Base58): ");
    if (seed_enc_ok) {
        printf("[%.*s] <-- Import this into Phantom/Solflare etc.\n", (int)b58_seed_size, b58_encoded_seed);
    } else {
        printf("[ENCODING FAILED - Needed buffer size: %zu]\n", b58_seed_size);
    }

    // Optional: Print Hex Seed for verification/debugging
    printf("Secret Seed (Hex)   : [");
    for(int i=0; i<32; ++i) printf("%02x", key_pair.seed[i]);
    printf("]\n");


    printf("--------------------------\n");
}


void vanity_run(config &vanity) {
	unsigned long long int  executions_total = 0;
	int  keys_found_total = 0;

	// Allocate host buffer for results from all GPUs
	std::vector<FoundKeyPair> host_found_keys(vanity.gpu_count * STOP_AFTER_KEYS_FOUND);
	std::vector<int> host_keys_found_index(vanity.gpu_count); // Stores count found per GPU per iteration

	printf("\nStarting vanity search for suffix '%s'\n", host_suffix); // Use host_suffix from config.h

    // Declare iter outside the loop if you need it after, but it's cleaner not to.
    // int iter = 0; // <-- Alternative, but less clean

	for (int iter = 0; iter < MAX_ITERATIONS && keys_found_total < STOP_AFTER_KEYS_FOUND; ++iter) { // 'iter' scope starts here
		auto start  = std::chrono::high_resolution_clock::now();

		unsigned long long int executions_this_iteration = 0;
		int keys_found_this_iteration_total = 0; // Keys found across all GPUs in this iteration

		// Reset device counters and launch kernels on all GPUs
		// ... (kernel launch loop - unchanged) ...
        for (int g = 0; g < vanity.gpu_count; ++g) {
			gpuErrchk(cudaSetDevice(g));
            // Reset device counters for this iteration
			gpuErrchk(cudaMemset(vanity.dev_keys_found_index[g], 0, sizeof(int)));
			gpuErrchk(cudaMemset(vanity.dev_executions_this_gpu[g], 0, sizeof(int)));
			// Launch kernel using stored parameters
			vanity_scan<<<vanity.maxActiveBlocks[g], vanity.blockSizes[g]>>>(
				vanity.states[g],
				vanity.dev_keys_found_index[g],
				vanity.dev_executions_this_gpu[g],
				vanity.dev_found_keys[g],
				STOP_AFTER_KEYS_FOUND
			);
			gpuErrchk(cudaPeekAtLastError());
		}


        // Wait for kernels to finish and process results from all GPUs
		// ... (result processing loop - unchanged) ...
        for (int g = 0; g < vanity.gpu_count; ++g) {
			gpuErrchk(cudaSetDevice(g));
            gpuErrchk(cudaDeviceSynchronize());
			int keys_found_this_gpu = 0;
			gpuErrchk(cudaMemcpy(&keys_found_this_gpu, vanity.dev_keys_found_index[g], sizeof(int), cudaMemcpyDeviceToHost));
            if (keys_found_this_gpu > STOP_AFTER_KEYS_FOUND) {
                fprintf(stderr, "Warning: GPU %d reported %d keys found, but buffer only holds %d. Clamping results count to %d.\n", g, keys_found_this_gpu, STOP_AFTER_KEYS_FOUND, STOP_AFTER_KEYS_FOUND);
                keys_found_this_gpu = STOP_AFTER_KEYS_FOUND;
            }
            keys_found_this_iteration_total += keys_found_this_gpu;
			host_keys_found_index[g] = keys_found_this_gpu;
			int executions_this_gpu_count = 0;
			gpuErrchk(cudaMemcpy(&executions_this_gpu_count, vanity.dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost));
			executions_this_iteration += (unsigned long long)executions_this_gpu_count * ATTEMPTS_PER_EXECUTION;
			if (keys_found_this_gpu > 0) {
                size_t host_buffer_offset = (size_t)g * STOP_AFTER_KEYS_FOUND;
				gpuErrchk(cudaMemcpy(host_found_keys.data() + host_buffer_offset,
				                     vanity.dev_found_keys[g],
				                     keys_found_this_gpu * sizeof(FoundKeyPair),
				                     cudaMemcpyDeviceToHost));
			}
		}

        auto finish = std::chrono::high_resolution_clock::now();

		executions_total += executions_this_iteration;

		// Print keys found in *this* iteration from the host buffer
		// int current_total_keys_before_iteration = keys_found_total; // <--- FIX 1: REMOVED THIS LINE
		if (keys_found_this_iteration_total > 0) {
            printf("-> Found %d key(s) this iteration.\n", keys_found_this_iteration_total);
            for (int g = 0; g < vanity.gpu_count; ++g) {
                size_t host_buffer_offset = (size_t)g * STOP_AFTER_KEYS_FOUND;
                for (int k = 0; k < host_keys_found_index[g]; ++k) {
                    if (keys_found_total < STOP_AFTER_KEYS_FOUND) {
                        print_found_key(host_found_keys[host_buffer_offset + k], g);
                        keys_found_total++;
                    } else {
                        break;
                    }
                }
                if (keys_found_total >= STOP_AFTER_KEYS_FOUND) break;
            }
        }

		// Print out performance Summary
		std::chrono::duration<double> elapsed = finish - start;
		double rate = (elapsed.count() > 0) ? (executions_this_iteration / elapsed.count()) : 0.0;
		printf("%s Iter %d | Found: %d | Speed: %.2f Mcps | Total Att: %llu | Elapsed: %.3fs\n",
			getTimeStr().c_str(),
			iter + 1, // 'iter' is still in scope here inside the loop
			keys_found_total,
			rate / 1.0e6,
			executions_total,
			elapsed.count()
		);

        // Check if we've found enough keys globally (inside the loop for early exit)
        if (keys_found_total >= STOP_AFTER_KEYS_FOUND) {
            printf("\nTarget key count (%d) reached. Finishing.\n", STOP_AFTER_KEYS_FOUND);
            break; // Exit the iteration loop
        }
	} // 'iter' scope ends here

    // --- FIX 2: Adjusted post-loop check ---
    // This code runs *after* the loop has finished (either by break or reaching MAX_ITERATIONS)
	// If the loop finished AND we haven't found enough keys, it must have hit the iteration limit.
	if (keys_found_total < STOP_AFTER_KEYS_FOUND) {
		printf("\nMaximum iterations (%d) reached. Finishing.\n", MAX_ITERATIONS);
	}
    // The 'else if' condition below was redundant because the first 'if' covers the only
    // way to exit the loop without enough keys. If enough keys were found, the 'break'
    // inside the loop was hit, and the message was already printed.
    /*
	else if (keys_found_total < STOP_AFTER_KEYS_FOUND) { // <-- This check is now redundant/unreachable logic based on the above 'if'
        printf("\nSearch loop finished unexpectedly. Found %d keys.\n", keys_found_total);
    }
    */
}

void vanity_cleanup(config &vanity) {
	printf("\nCleaning up GPU resources...\n");
	for (int g = 0; g < vanity.gpu_count; ++g) {
        // It's good practice to set device context before freeing its memory
        // although cudaFree might work regardless on modern drivers.
		gpuErrchk(cudaSetDevice(g));
		// Free all allocated device memory
		if (vanity.states[g]) gpuErrchk(cudaFree(vanity.states[g]));
		if (vanity.dev_keys_found_index[g]) gpuErrchk(cudaFree(vanity.dev_keys_found_index[g]));
		if (vanity.dev_executions_this_gpu[g]) gpuErrchk(cudaFree(vanity.dev_executions_this_gpu[g]));
		if (vanity.dev_found_keys[g]) gpuErrchk(cudaFree(vanity.dev_found_keys[g]));
	}
    // Reset device context to default (optional but good practice)
    cudaSetDevice(0);
	printf("Cleanup complete.\n");
}


/* -- CUDA Vanity Functions ------------------------------------------------- */

// Initialize CURAND states for N threads
// N is the total number of threads across all blocks for this GPU
void __global__ vanity_init(unsigned long long int* rseed, curandState* state, int N) {
	// Calculate global thread ID
    int id = threadIdx.x + blockIdx.x * blockDim.x;

	// Check array bounds before initializing state
	if (id < N) {
        // Initialize the state: seed, sequence number, offset
		curand_init(*rseed + id, /* sequence */ id, /* offset */ 0, &state[id]);
	}
}

// Main kernel: Generate keys and check for suffix match
// Note: Removed unused 'gpu' pointer from signature
void __global__ vanity_scan(curandState* state, int* keys_found_index, int* exec_count, FoundKeyPair* results_buffer, int max_results) {
	// Calculate global thread ID
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Atomically increment the execution counter for this *thread launch*
    // Note: This counts thread launches, not key checks yet.
    // The host multiplies this by ATTEMPTS_PER_EXECUTION later.
    atomicAdd(exec_count, 1);

	curandState localState = state[id]; // Load state into local thread registers

	// Local Kernel State - allocate on thread's stack
	unsigned char seed[32];             // 32-byte seed for key generation
	unsigned char publick[32];          // Public key output
	unsigned char privatek[64];         // Expanded private key output (64 bytes) - still needed by ed25519_create_keypair
	char b58_encoded_key[64];           // Buffer for Base58 encoded public key (max ~44 chars + null)
	size_t b58_pub_size;                // Size of encoded public key

	// Perform multiple attempts within a single thread execution
	for (int i = 0; i < ATTEMPTS_PER_EXECUTION; i++) {
		// 1. Generate a random 32-byte seed for this attempt
        // Use curand() to fill the seed buffer.
        // Note: curand() returns unsigned int (4 bytes).
        unsigned int* seed_as_uint = (unsigned int*)seed;
        // Check alignment if necessary, but casting usually works.
        #pragma unroll // Help compiler optimize this loop
        for (int k = 0; k < 8; ++k) { // Generate 8 * 4 = 32 bytes
             seed_as_uint[k] = curand(&localState);
        }

		// 2. Derive the public key and the 64-byte expanded private key from the random seed
        // Assumes ed25519_create_keypair takes seed (in) and produces publick, privatek (out)
		ed25519_create_keypair(publick, privatek, seed);

		// 3. Encode public key to Base58
		b58_pub_size = sizeof(b58_encoded_key); // Pass buffer size IN
		bool enc_pub_ok = b58enc(b58_encoded_key, &b58_pub_size, publick, 32);
        // b58_pub_size now holds the encoded length OUT (excluding null)

		// 4. Check for suffix match using precomputed length from config.h
		bool match = false;
		if (enc_pub_ok && b58_pub_size >= suffix_length) {
			match = true; // Assume match until proven otherwise
			// Compare the *end* of the Base58 string with the device_suffix
            // device_suffix and suffix_length must be defined (e.g., in config.h)
            #pragma unroll // Help compiler optimize this small fixed loop
			for (int j = 0; j < suffix_length; j++) {
                // Compare character by character from the end backwards
				if (b58_encoded_key[b58_pub_size - suffix_length + j] != device_suffix[j]) {
					match = false;
					break; // Mismatch found, no need to check further
				}
			}
		}

		// 5. If match found, store result in the output buffer
		if (match) {
			// Atomically get the next available index in the results buffer for this GPU
			int result_idx = atomicAdd(keys_found_index, 1);

			// Ensure we don't write out of bounds of the results buffer for this GPU
			if (result_idx < max_results) {
				// Copy keys to the results buffer at the obtained index
				// Use memcpy for potentially better performance on aligned data
                memcpy(results_buffer[result_idx].public_key, publick, 32);
                // Store the ORIGINAL SEED, not the expanded private key
                memcpy(results_buffer[result_idx].seed, seed, 32);
			}
            // If result_idx >= max_results, the key is found but cannot be stored.
            // The host will detect this discrepancy (keys_found_index > max_results) and warn.
		}

		// No seed increment needed; we generate a fresh random seed each loop iteration.
	}

	// Save the potentially updated PRNG state back to global memory
	state[id] = localState;
}


// Base58 encoding function (device version)
// Made input data const uint8_t* for correctness
// Increased buffer size slightly just in case, though 128 was likely ok.
bool __device__ b58enc(char* b58, size_t* b58sz, const uint8_t* data, size_t binsz)
{
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
	const uint8_t* bin = data; // Use const pointer internally
	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;

	// Count leading zeros
	while (zcount < binsz && !bin[zcount])
		++zcount;

	// Calculate estimate for result size
	size = (binsz - zcount) * 138 / 100 + 1;
	// Use a stack buffer large enough for expected inputs (e.g., 64 bytes input -> ~88 chars output)
	uint8_t buf[150]; // Increased size slightly for safety margin

    // Check if estimated size exceeds our stack buffer. Should not happen for typical key sizes.
	if (size > sizeof(buf)) {
        // Indicate error: internal buffer too small (this is a programming error if it happens)
		*b58sz = 0; // Indicate error
		return false;
	}

    // Initialize buffer. Standard memset is often optimized by NVCC for device code.
	memset(buf, 0, sizeof(buf)); // Zero the whole buffer for safety, not just 'size' bytes

    // Conversion loop
	for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
	{
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
		{
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j && (carry || j > high) ) { // Check j before loop condition might access buf[-1]
				// Avoid infinite loop/wrap-around if carry remains but j is 0
                // This condition might indicate an issue if reached.
				break;
			}
            // Prevent j from wrapping around if loop condition is met at j=0
            if (!j) break;
		}
	}

    // Find first non-zero digit in result buffer
	for (j = 0; j < size && !buf[j]; ++j);

    // Calculate required output size (+1 for null terminator)
    size_t out_size_needed = zcount + size - j + 1;

    // Check if the provided output buffer `b58` is large enough
	if (*b58sz < out_size_needed) // Compare required vs available (passed in *b58sz)
	{
		// Output buffer too small. Return required size.
		*b58sz = out_size_needed; // Inform caller of needed size
		return false;
	}

    // Fill output buffer: leading '1's for leading zeros in input
	if (zcount)
		memset(b58, '1', zcount);
    // Fill output buffer: converted digits
	for (i = zcount; j < size; ++i, ++j)
		b58[i] = b58digits_ordered[buf[j]];
    // Null-terminate the output string
	b58[i] = '\0';
    // Update the size parameter (*b58sz) to reflect the actual length written (excluding null)
	*b58sz = i;

	return true;
}

// Host-side Base58 encoding function (copied and adapted from device version)
// Uses standard C/C++ types and functions.
bool host_b58enc(char* b58, size_t* b58sz, const unsigned char* data, size_t binsz)
{
    const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    const uint8_t* bin = data; // Use const internally
    long long carry; // Use larger type for carry on host potentially? int should be fine.
    size_t i, j, high, zcount = 0;
    size_t size;

    // Count leading zeros
    while (zcount < binsz && !bin[zcount])
        ++zcount;

    // Calculate estimate for result size
    size = (binsz - zcount) * 138 / 100 + 1;
    // Use a stack buffer large enough for expected inputs (e.g., 64 bytes input -> ~88 chars output)
    uint8_t buf[150]; // Matches device version size

    // Check if estimated size exceeds our stack buffer. Should not happen for typical key sizes.
	if (size > sizeof(buf)) {
		*b58sz = size + zcount + 1; // Return required size guess
		return false;
	}

    memset(buf, 0, sizeof(buf)); // Zero the buffer

    // Conversion loop
	for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
	{
		// Inner loop needs careful boundary checks
		for (carry = bin[i], j = size - 1; ; --j) // Condition check inside loop
		{
			carry += 256LL * buf[j]; // Use LL suffix for literal if using long long carry
			buf[j] = carry % 58;
			carry /= 58;

            // Break conditions:
            // 1. Carry is zero AND we've processed up to the 'high' water mark
            if (carry == 0 && j <= high) break;
            // 2. Prevent j from going below 0
            if (j == 0) break;
		}
	}

    // Find first non-zero digit in result buffer
	for (j = 0; j < size && !buf[j]; ++j);

    // Calculate required output size (+1 for null terminator)
    size_t out_size_needed = zcount + size - j + 1;

    // Check if the provided output buffer `b58` is large enough
	if (*b58sz < out_size_needed)
	{
		*b58sz = out_size_needed; // Inform caller of needed size
		return false;
	}

    // Fill output buffer: leading '1's
	if (zcount)
		memset(b58, '1', zcount);
    // Fill output buffer: converted digits
	for (i = zcount; j < size; ++i, ++j) {
        // Added bounds check for safety, though buf[j] should be < 58
        if (buf[j] >= 58) { /* Handle error or assert */ return false; }
		b58[i] = b58digits_ordered[buf[j]];
    }
    // Null-terminate
	b58[i] = '\0';
    // Update the size parameter (*b58sz) to reflect the actual length written (excluding null)
	*b58sz = i;

	return true;
}