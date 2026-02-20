#pragma once
inline constexpr int kFRAMERATE{60};
inline constexpr int kROUNDS{10000};
// Higher dimensional parameters
inline constexpr int kTREE_MEDIAN_SAMPLE{30}; // could profile
inline constexpr int kLEAF_SIZE{16};          // double once using floats to fit SIMD registers better, could profile
inline constexpr double kPRUNE_EPS{0.01};     // for approximate KD tree pruning
inline constexpr double kAPPROX_PRUNE_FACTOR{1.0 / ((1.0 + kPRUNE_EPS) * (1.0 + kPRUNE_EPS))};
