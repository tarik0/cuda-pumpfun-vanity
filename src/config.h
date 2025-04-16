#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 100000;
static int const STOP_AFTER_KEYS_FOUND = 100;

// how many times a gpu thread generates a public key in one go
// Increased significantly for hypothetical RTX 5090 - tune based on observed performance.
// Reduced from 100M to 10M to prevent TDR timeouts, may need further tuning.
__constant__ const int ATTEMPTS_PER_EXECUTION = 10000000; // Was 100,000,000

// __device__ const int MAX_PATTERNS = 10; // Removed - Unused

// pump.fun suffix

// For the device kernel
// __device__ static char const *suffix = "pump"; // OLD definition
__constant__ const char device_suffix[] = "pump"; // Use __constant__ for read-only kernel data
__device__ const int suffix_length = 4; // Precomputed length of "pump" - Kept __device__ as kernel uses it

// For the host code
const char host_suffix[] = "pump";

#endif
