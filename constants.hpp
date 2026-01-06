#pragma once
// parameters
inline constexpr int n_birds{200};
inline constexpr int M{7};                   // bird set size,        7
inline constexpr double delay{0.1};          // delay,                0.1
inline constexpr double pt{700};             // persistence time,     700
inline constexpr double pd{20};              // persistence distance, 20
inline constexpr double rt{800};             // refractory time,      800
inline constexpr double rep{0.01};           // repulsion force,      2.5
inline constexpr double ali{0.01};           // alignment force,      3
inline constexpr double att{100};            // attraction force,     0.01
inline constexpr double timestep{0.1};       //                       0.1
inline constexpr double probability{0.0002}; //                       0.0002
inline constexpr double epsilon{0.0001};     // std::numeric_limits<double>::epsilon() for machine epsilon, 0.0001

// window
inline constexpr float width{1100};
inline constexpr float height{400};
inline constexpr float bound_offset{0.1};
inline constexpr int starting_interval{200}; // bound which birds are initialized
inline constexpr int framerate{60};