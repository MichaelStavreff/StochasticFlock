#pragma once
// parameters
inline constexpr int kN_BIRDS{100};
inline constexpr int kM{6};                   // bird set size,        7
inline constexpr double kDELAY{0.1};          // delay,                0.1
inline constexpr double kPT{60};              // persistence time,     700
inline constexpr double kPD{150};             // persistence distance, 20
inline constexpr double kRT{60};              // refractory time,      800
inline constexpr double kREP{0.5};            // repulsion force,      2.5
inline constexpr double kALI{0.05};           // alignment force,      3
inline constexpr double kATT{0.15};           // attraction force,     0.01
inline constexpr double kTIMESTEP{0.05};      //                       0.1
inline constexpr double kPROBABILITY{0.0002}; //                       0.0002
inline constexpr double kEPSILON{0.1};        // std::numeric_limits<double>::epsilon() for machine epsilon, 0.0001
inline constexpr int kBUFFER_CYCLES{static_cast<int>(kDELAY / kTIMESTEP)};

inline constexpr int kFRAMERATE{60};
inline constexpr int kROUNDS{10000};
// Higher dimensional parameters
inline constexpr int kTREE_MEDIAN_SAMPLE{30}; // could profile
inline constexpr int kLEAF_SIZE{16};          // double once using floats to fit SIMD registers better, could profile
inline constexpr double kPRUNE_EPS{0.01};     // for approximate KD tree pruning
inline constexpr double kAPPROX_PRUNE_FACTOR{1.0 / ((1.0 + kPRUNE_EPS) * (1.0 + kPRUNE_EPS))}; // Precompute this!
// 1D window
inline constexpr float kWIDTH{2200};
inline constexpr float kHEIGHT{400};
inline constexpr float kBOUND_OFFSET{0.1};    // purely visual "height" of 1D line
inline constexpr int kSTARTING_INTERVAL{200}; // bound which birds are initialized
// 2D window
inline constexpr float kWIDTH_2D{1200};
inline constexpr float kHEIGHT_2D{900};
inline constexpr float kBOX_SIZE{200}; // starting interval in 2D