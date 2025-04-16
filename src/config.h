#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 100000;
static int const STOP_AFTER_KEYS_FOUND = 100;

// how many times a gpu thread generates a public key in one go
__constant__ const int ATTEMPTS_PER_EXECUTION = 300000; // 300K

// __device__ const int MAX_PATTERNS = 10; // Removed - Unused

// pump.fun suffix

// For the device kernel
// __device__ static char const *suffix = "pump"; // OLD definition
__constant__ const char device_suffix[] = "pump"; // Use __constant__ for read-only kernel data
__device__ const int suffix_length = 4; // Precomputed length of "pump" - Kept __device__ as kernel uses it

// For the host code
const char host_suffix[] = "pump";

#endif
