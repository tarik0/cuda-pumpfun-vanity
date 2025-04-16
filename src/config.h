#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 100000;
static int const STOP_AFTER_KEYS_FOUND = 100;

// how many times a gpu thread generates a public key in one go
// Increased significantly for hypothetical RTX 5090 - tune based on observed performance.
__device__ const int ATTEMPTS_PER_EXECUTION = 100000000; // Was 1,000,000

__device__ const int MAX_PATTERNS = 10;

// pump.fun suffix

__device__ static char const *suffix = "pump";
__device__ const int suffix_length = 4; // Precomputed length of "pump"

#endif
