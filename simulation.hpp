#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "constants.hpp"

// rowmajor mistake, need column for SIMD
//  avoid recursion for  KD tree, extra overhead for additional stack frames
//  pre-allocate memory for tree algorithm
//  revisit nearest neighbors for 1D, pointer walk better?
//

struct Simulation1d
{
    int steps_back{};
    int delayed_steps{};

    std::mt19937 seed;
    double rate{probability / 0.1};
    std::bernoulli_distribution status{rate * timestep};

    Eigen::Matrix<double, n_birds + n_birds * buffer_cycles, 3, Eigen::RowMajor> full_states;
    Eigen::Map<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>> states;
    Eigen::Map<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>> buffer;
    Eigen::Vector<double, n_birds> y_states;
    // we store timers apart from states to not clutter CPU registers/SIMD
    Eigen::Matrix<double, n_birds + n_birds * buffer_cycles, 2, Eigen::RowMajor> timers;
    Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>> timer_states;
    Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>> timer_buffer;

    // nearest neighbor buffer
    Eigen::Vector<int, M> neighbors;
    // flock acceleration buffers
    double target_val;
    std::vector<int> idx;

    Simulation1d()
        : states(full_states.topRows(n_birds).data()), buffer(full_states.topRows(n_birds).data()),
          timer_states(timers.topRows(n_birds).data()), timer_buffer(timers.topRows(n_birds).data())
    {
        seed.seed(std::random_device{}());
        auto positions{generate_initial_state()};
        y_states = positions.col(1);
        full_states.setZero();
        timers.setZero();
        full_states.col(0) = positions.col(0).replicate(1 + buffer_cycles, 1);

        idx.resize(n_birds);
        // initial offset of maps
        double *new_address = states.data() + (1 * n_birds * 3);
        new (&buffer) Eigen::Map<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>>(new_address, n_birds, 3);

        double *new_address_timers = timers.data() + (1 * n_birds * 2);
        new (&timer_buffer)
            Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>>(new_address_timers, n_birds, 2);
    }

    Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor> generate_initial_state()
    {
        std::uniform_real_distribution<double> starting_x{-starting_interval / 2, starting_interval / 2};
        std::uniform_real_distribution<double> starting_y{height * (0.5f - bound_offset),
                                                          height * (0.5f + bound_offset)};
        Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor> starting_states;
        starting_states.col(0) = starting_states.col(0).NullaryExpr([&]() { return starting_x(seed); });
        starting_states.col(1) = starting_states.col(1).NullaryExpr([&]() { return starting_y(seed); });
        return starting_states;
    }

    Eigen::Vector<int, M> nearest_M_idx(
        const Eigen::Ref<const Eigen::Vector<double, n_birds>, 0, Eigen::OuterStride<3>> positions, int target_idx)
    {
        target_val = positions(target_idx);
        std::iota(idx.begin(), idx.end(), 0);

        std::nth_element(idx.begin(), idx.begin() + M, idx.end(), [&](int i, int j) {
            return std::abs(positions(i) - target_val) < std::abs(positions(j) - target_val);
        });

        for (int i = 0; i < M; ++i)
        {
            neighbors(i) = idx[i];
        }
        std::sort(neighbors.data(), neighbors.data() + neighbors.size());
        return neighbors;
    }

    Eigen::Vector<double, 2 * n_birds> flock_acceleration(
        Eigen::Ref<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>> states)
    {
        // Eigen::internal::set_is_malloc_allowed(false);
        Eigen::Vector<double, n_birds> acceleration_vec;
        Eigen::Vector<double, n_birds> closest_neighbors;
        Eigen::Vector<double, 2 * n_birds> packed_result;

        auto positions{states.col(0)};
        auto velocities{states.col(1)};
        auto status{states.col(2).array()};
        double cm{ali / M};
        for (int i = 0; i < n_birds; ++i)
        {
            Eigen::Vector<int, M> m_set_indices{nearest_M_idx(
                positions,
                i)}; // fix this looped allocation by having the function take by reference and change (non const)
            closest_neighbors(i) = m_set_indices(0);
            auto neighbor_positions{positions(m_set_indices).array()};
            auto neighbor_velocities{velocities(m_set_indices).array()};
            auto pos_diff{neighbor_positions - positions(i)};
            auto vel_diff{neighbor_velocities - velocities(i)};
            auto a_sum_1{((pos_diff.array()) / ((pos_diff).array().abs().square() + epsilon))};
            auto a_sum_2{(cm * (vel_diff) + att * (pos_diff))};
            double acceleration{-rep * a_sum_1.sum() + (1.0 - status(i)) * a_sum_2.sum()};
            acceleration_vec(i) = acceleration;
        }
        packed_result << acceleration_vec,
            closest_neighbors; // packing closest neighbors for persistence distance tracking
        // Eigen::internal::set_is_malloc_allowed(true);
        return packed_result;
    }
    void shift_back() // shifts both maps, remapping is not a heap allocation!
    {
        steps_back = (steps_back + 1) % (buffer_cycles + 1); // circular modulus
        double *new_address_states = full_states.data() + (steps_back * n_birds * 3);
        new (&states) Eigen::Map<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>>(new_address_states, n_birds, 3);

        double *new_address_timers = timers.data() + (steps_back * n_birds * 2);
        new (&timer_states)
            Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>>(new_address_timers, n_birds, 2);

        int delayed_steps = (steps_back + 1) % (buffer_cycles + 1);
        double *new_address_buffer = full_states.data() + delayed_steps * n_birds * 3;
        new (&buffer) Eigen::Map<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>>(new_address_buffer, n_birds, 3);

        double *new_address_timers_delay = timers.data() + delayed_steps * n_birds * 2;
        new (&timer_buffer)
            Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>>(new_address_timers_delay, n_birds, 2);
    }

    void update_state(Eigen::Ref<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>> states)
    {
        auto acceleration_return{flock_acceleration(buffer)};

        for (int i = 0; i < n_birds; ++i)
        {
            // we use < or > 0.5 instead of == 1 to avoid floating point issues

            if (states.col(2)[i] > 0.5 &&
                timer_buffer.col(0)[i] < pt) // check if leaders should expire, assign refractory period
            {
                buffer.col(2)[i] = 0;
                timer_buffer.col(1)[i] = rt;
            }
            if (states.col(2)[i] > 0.5 && timer_states.col(0)[i] > 0) // decremet leaders' persistence time
                timer_buffer.col(0)[i] -= timestep;
            if (states.col(2)[i] < 0.5 && timer_states.col(1)[i] > 0 &&
                timer_states.col(1)[i] != rt) // decrement followers' refractory time who weren't just demoted
                timer_buffer.col(1)[i] -= timestep;
            if (states.col(2)[i] < 0.5 &&
                timer_states.col(1)[i] == 0) // followers without refractory time roll for leader
                buffer.col(2)[i] = status(seed);
            if (buffer.col(2)[i] > 0.5 &&
                states.col(2)[i] < 0.5) // assign new leaders persistence time if they werent one in previous step
                timer_buffer.col(0)[i] = pt;
            if (states.col(2)[i] > 0.5) // propagate leaders to next state last
                buffer.col(2)[i] = 1;
            if (states.col(2)[i] > 0.5 && acceleration_return(n_birds + i) > pd)
            { // demote leaders which stray too far
                buffer.col(2)[i] = 0;
                timer_buffer.col(1)[i] = rt;
            }
        }
        buffer.col(1) = states.col(1) + acceleration_return(Eigen::seq(0, n_birds - 1)) *
                                            timestep; //"future" position, will become present after shift
        buffer.col(0) = states.col(0) + states.col(1) * timestep;
        shift_back();
    }
};

// struct Simulation2d
// {
//     int steps_back{};
//     int delayed_steps{};

//     std::mt19937 seed;
//     double rate{probability / 0.1};
//     std::bernoulli_distribution status{rate * timestep};

//     Eigen::Matrix<double, n_birds + n_birds * buffer_cycles, 4, Eigen::RowMajor> full_states;
//     Eigen::Map<Eigen::Matrix<double, n_birds, 4, Eigen::RowMajor>> states;
//     Eigen::Map<Eigen::Matrix<double, n_birds, 4, Eigen::RowMajor>> buffer;
//     // we store timers apart from states to not clutter CPU registers/SIMD
//     Eigen::Matrix<double, n_birds + n_birds * buffer_cycles, 2, Eigen::RowMajor> timers;
//     Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>> timer_states;
//     Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>> timer_buffer;

//     // memory pool for KNN tree nodes
//     //  2 columns for coords, 2 for left/right child
//     Eigen::Matrix<double, n_birds, 4, Eigen::RowMajor> full_tree;

//     Simulation2d()
//         : states(full_states.topRows(n_birds).data()), buffer(full_states.topRows(n_birds).data()),
//           timer_states(timers.topRows(n_birds).data()), timer_buffer(timers.topRows(n_birds).data())
//     {
//         seed.seed(std::random_device{}());
//         auto positions{generate_initial_state()};
//         full_states.setZero();
//         timers.setZero();
//         full_states.col(0) = positions.col(0).replicate(1 + buffer_cycles, 1);
//         // initial offset of maps
//         double *new_address = states.data() + (1 * n_birds * 4);
//         new (&buffer) Eigen::Map<Eigen::Matrix<double, n_birds, 4, Eigen::RowMajor>>(new_address, n_birds, 4);

//         double *new_address_timers = timers.data() + (1 * n_birds * 2);
//         new (&timer_buffer)
//             Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>>(new_address_timers, n_birds, 2);
//     }

//     Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor> generate_initial_state()
//     {
//         std::uniform_real_distribution<double> starting_x{-starting_interval / 2, starting_interval / 2};
//         std::uniform_real_distribution<double> starting_y{height * (0.5f - bound_offset),
//                                                           height * (0.5f + bound_offset)};
//         Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor> starting_states;
//         starting_states.col(0) = starting_states.col(0).NullaryExpr([&]() { return starting_x(seed); });
//         starting_states.col(1) = starting_states.col(1).NullaryExpr([&]() { return starting_y(seed); });
//         return starting_states;
//     }

//     Eigen::Vector<double, M> nearest_M(
//         const Eigen::Ref<const Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>> positions, int target_idx)
//     {
//         Eigen::Matrix<double, M, 2> neighbors;
//         auto target_val = positions.row(target_idx);
//         std::vector<int> idx(positions.size());
//         std::iota(idx.begin(), idx.end(), 0);

//         std::nth_element(idx.begin(), idx.begin() + M, idx.end(), [&](int i, int j) {
//             return std::abs(positions(i) - target_val) < std::abs(positions(j) - target_val);
//         });

//         for (int i = 0; i < M; ++i)
//         {
//             neighbors(i) = idx[i];
//         }
//         std::sort(neighbors.data(), neighbors.data() + neighbors.size());
//         return neighbors;
//     }

//     Eigen::Matrix<double, 2 * n_birds, 2> flock_acceleration(
//         Eigen::Ref<Eigen::Matrix<double, n_birds, 4, Eigen::RowMajor>> states)
//     {
//         Eigen::Matrix<double, n_birds, 2> acceleration_mat;
//         Eigen::Vector<double, n_birds> closest_neighbors;
//         Eigen::Matrix<double, 2 * n_birds, 2> packed_result;

//         auto positions{states.col(0)};
//         auto velocities{states.col(1)};
//         auto status{states.col(2).array()};
//         double cm{ali / M};
//         for (int i = 0; i < n_birds; ++i)
//         {
//             Eigen::Vector<double, M> m_set_indices{nearest_M(positions, i)};
//             closest_neighbors(i) = m_set_indices(0);
//             auto neighbor_positions{positions(m_set_indices).array()};
//             auto neighbor_velocities{velocities(m_set_indices).array()};
//             auto pos_diff{neighbor_positions - positions(i)};
//             auto vel_diff{neighbor_velocities - velocities(i)};
//             auto a_sum_1{((pos_diff.array()) / ((pos_diff).array().abs().square() + epsilon))};
//             auto a_sum_2{(cm * (vel_diff) + att * (pos_diff))};
//             double acceleration{-rep * a_sum_1.sum() + (1.0 - status(i)) * a_sum_2.sum()};
//             acceleration_mat(i) = acceleration;
//         }
//         packed_result << acceleration_mat,
//             closest_neighbors; // packing closest neighbors for persistence distance tracking
//         return packed_result;
//     }
//     void shift_back() // shifts both maps
//     {
//         steps_back = (steps_back + 1) % (buffer_cycles + 1); // circular modulus
//         double *new_address_states = full_states.data() + (steps_back * n_birds * 3);
//         new (&states) Eigen::Map<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>>(new_address_states, n_birds, 3);

//         double *new_address_timers = timers.data() + (steps_back * n_birds * 2);
//         new (&timer_states)
//             Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>>(new_address_timers, n_birds, 2);

//         int delayed_steps = (steps_back + 1) % (buffer_cycles + 1);
//         double *new_address_buffer = full_states.data() + delayed_steps * n_birds * 3;
//         new (&buffer) Eigen::Map<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>>(new_address_buffer, n_birds, 3);

//         double *new_address_timers_delay = timers.data() + delayed_steps * n_birds * 2;
//         new (&timer_buffer)
//             Eigen::Map<Eigen::Matrix<double, n_birds, 2, Eigen::RowMajor>>(new_address_timers_delay, n_birds, 2);
//     }

//     void update_state(Eigen::Ref<Eigen::Matrix<double, n_birds, 3, Eigen::RowMajor>> states)
//     {
//         auto acceleration_return{flock_acceleration(buffer)};

//         for (int i = 0; i < n_birds; ++i)
//         {
//             // we use < or > 0.5 instead of == 1 to avoid floating point issues

//             if (states.col(2)[i] > 0.5 &&
//                 timer_buffer.col(0)[i] < pt) // check if leaders should expire, assign refractory period
//             {
//                 buffer.col(2)[i] = 0;
//                 timer_buffer.col(1)[i] = rt;
//             }
//             if (states.col(2)[i] > 0.5 && timer_states.col(0)[i] > 0) // decremet leaders' persistence time
//                 timer_buffer.col(0)[i] -= timestep;
//             if (states.col(2)[i] < 0.5 && timer_states.col(1)[i] > 0 &&
//                 timer_states.col(1)[i] != rt) // decrement followers' refractory time who weren't just demoted
//                 timer_buffer.col(1)[i] -= timestep;
//             if (states.col(2)[i] < 0.5 &&
//                 timer_states.col(1)[i] == 0) // followers without refractory time roll for leader
//                 buffer.col(2)[i] = status(seed);
//             if (buffer.col(2)[i] > 0.5 &&
//                 states.col(2)[i] < 0.5) // assign new leaders persistence time if they werent one in previous
//                 step timer_buffer.col(0)[i] = pt;
//             if (states.col(2)[i] > 0.5) // propagate leaders to next state last
//                 buffer.col(2)[i] = 1;
//             if (states.col(2)[i] > 0.5 && acceleration_return(n_birds + i) > pd)
//             { // demote leaders which stray too far
//                 buffer.col(2)[i] = 0;
//                 timer_buffer.col(1)[i] = rt;
//             }
//         }
//         buffer.col(1) = states.col(1) + acceleration_return(Eigen::seq(0, n_birds - 1)) *
//                                             timestep; //"future" position, will become present after shift
//         buffer.col(0) = states.col(0) + states.col(1) * timestep;
//         shift_back();
//     }
// };