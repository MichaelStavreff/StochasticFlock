#pragma once
#include <Eigen/Dense>
#include <algorithm>
// #include <deque>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "constants.hpp"
using namespace Eigen;
using namespace std;

struct Simulation1d
{
    int dimensions{};
    static constexpr int buffer_cycles{static_cast<int>(delay / timestep)};
    int steps_back{};
    int delayed_steps{};
    int rounds{};

    mt19937 seed;
    double rate{probability / 0.1};
    bernoulli_distribution status{rate * timestep};

    Matrix<double, n_birds + n_birds * buffer_cycles, 3, RowMajor> full_states;
    Map<Matrix<double, n_birds, 3, RowMajor>> states;
    Map<Matrix<double, n_birds, 3, RowMajor>> buffer;
    Vector<double, n_birds> y_states;
    // we store timers apart from states to not clutter CPU registers/SIMD
    Matrix<double, n_birds + n_birds * buffer_cycles, 2, RowMajor> timers;
    Map<Matrix<double, n_birds, 2, RowMajor>> timer_states;
    Map<Matrix<double, n_birds, 2, RowMajor>> timer_buffer;

    Simulation1d()
        : states(full_states.topRows(n_birds).data()), buffer(full_states.topRows(n_birds).data()),
          timer_states(timers.topRows(n_birds).data()), timer_buffer(timers.topRows(n_birds).data())
    {
        seed.seed(random_device{}());
        auto positions{generate_initial_state()};
        y_states = positions.col(1);
        full_states.setZero();
        timers.setZero();
        full_states.col(0) = positions.col(0).replicate(1 + buffer_cycles, 1);
        // initial offset of maps
        double *new_address = states.data() + (1 * n_birds * 3);
        new (&buffer) Map<Matrix<double, n_birds, 3, RowMajor>>(new_address, n_birds, 3);

        double *new_address_timers = timers.data() + (1 * n_birds * 2);
        new (&timer_buffer) Map<Matrix<double, n_birds, 2, RowMajor>>(new_address_timers, n_birds, 2);
    }

    Matrix<double, n_birds, 2, RowMajor> generate_initial_state()
    {
        uniform_real_distribution<double> starting_x{-starting_interval / 2, starting_interval / 2};
        uniform_real_distribution<double> starting_y{height * (0.5f - bound_offset), height * (0.5f + bound_offset)};
        Matrix<double, n_birds, 2, RowMajor> starting_states;
        starting_states.col(0) = starting_states.col(0).NullaryExpr([&]() { return starting_x(seed); });
        starting_states.col(1) = starting_states.col(1).NullaryExpr([&]() { return starting_y(seed); });
        return starting_states;
    }

    Vector<double, M> nearest_M_indices(const Ref<const Vector<double, n_birds>, 0, OuterStride<>> positions,
                                        int target_idx)
    // ^ sacrificing locality since passing vector extracted from rowmajor matrix, thus CPU must skips addresses to
    // retrieve vector element
    {
        Vector<double, M> neighbors;

        vector<int> sorted_indices(positions.size());
        iota(sorted_indices.begin(), sorted_indices.end(), 0);
        sort(sorted_indices.begin(), sorted_indices.end(), [&](int i, int j) { return positions(i) < positions(j); });

        auto it{find(sorted_indices.begin(), sorted_indices.end(), target_idx)};

        auto L{distance(sorted_indices.begin(), it) - 1};
        auto R{distance(sorted_indices.begin(), it) + 1};

        double target_val{positions(target_idx)};
        int count{0};

        while (count < M)
        {
            bool can_go_left{(L >= 0)};
            bool can_go_right{(R < ssize(sorted_indices))};

            if (can_go_left && can_go_right)
            {
                if (abs(positions(sorted_indices[L]) - target_val) <= abs(positions(sorted_indices[R]) - target_val))
                {
                    neighbors(count++) = sorted_indices[L--];
                }
                else
                {
                    neighbors(count++) = sorted_indices[R++];
                }
            }
            else if (can_go_left)
            {
                neighbors(count++) = sorted_indices[L--];
            }
            else if (can_go_right)
            {
                neighbors(count++) = sorted_indices[R++];
            }
            else
            {
                break; // Should not happen if n_birds > M
            }
        }
        return neighbors;
    }

    Vector<double, 2 * n_birds> flock_acceleration(Ref<Matrix<double, n_birds, 3, RowMajor>> states)
    {
        Vector<double, n_birds> acceleration_vec;
        Vector<double, n_birds> closest_neighbors;
        Vector<double, 2 * n_birds> packed_result;

        auto positions{states.col(0)};
        auto velocities{states.col(1)};
        auto status{states.col(2).array()};
        double cm{ali / M};
        for (int i = 0; i < n_birds; ++i)
        {
            Vector<double, M> m_set_indices{nearest_M_indices(positions, i)};
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
        return packed_result;
    }
    void shift_back() // shifts both maps
    {
        steps_back = (steps_back + 1) % (buffer_cycles + 1); // circular modulus
        double *new_address_states = full_states.data() + (steps_back * n_birds * 3);
        new (&states) Map<Matrix<double, n_birds, 3, RowMajor>>(new_address_states, n_birds, 3);

        double *new_address_timers = timers.data() + (steps_back * n_birds * 2);
        new (&timer_states) Map<Matrix<double, n_birds, 2, RowMajor>>(new_address_timers, n_birds, 2);

        int delayed_steps = (steps_back + 1) % (buffer_cycles + 1);
        double *new_address_buffer = full_states.data() + delayed_steps * n_birds * 3;
        new (&buffer) Map<Matrix<double, n_birds, 3, RowMajor>>(new_address_buffer, n_birds, 3);

        double *new_address_timers_delay = timers.data() + delayed_steps * n_birds * 2;
        new (&timer_buffer) Map<Matrix<double, n_birds, 2, RowMajor>>(new_address_timers_delay, n_birds, 2);
    }

    void update_state(Ref<Matrix<double, n_birds, 3, RowMajor>> states)
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
        buffer.col(1) = states.col(1) + acceleration_return(seq(0, n_birds - 1)) *
                                            timestep; //"future" position, will become present after shift
        buffer.col(0) = states.col(0) + states.col(1) * timestep;
        shift_back();
    }
};
