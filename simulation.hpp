#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <ostream> //endl
#include <random>

#include "constants.hpp"

//  avoid recursion for  KD tree, extra overhead for additional stack frames
//  pre-allocate memory for tree algorithm
//  revisit nearest neighbors for 1D, pointer walk better?
// cmake
// evaluate where to include floats or doubles (during calculations/cast back to float for storage)
// verify SIMD/cache misses

class Simulation1d
{
  public:
    Eigen::Matrix<double, kN_BIRDS, 3 * (1 + kBUFFER_CYCLES)> full_states;
    Eigen::Matrix<double, kN_BIRDS, 2 * (1 + kBUFFER_CYCLES)> timers;

  private:
    Eigen::Vector<double, 2 * kN_BIRDS> packed_result_;
    Eigen::Vector<double, kN_BIRDS> acceleration_vec_;
    Eigen::Vector<double, kN_BIRDS> closest_neighbors_;
    Eigen::Vector<int, kN_BIRDS> idx_;

  public:
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 3>> states;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 3>> buffer;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>> timer_states;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>> timer_buffer;
    Eigen::Vector<double, kN_BIRDS> y_states;

  private:
    Eigen::Vector<int, kM> neighbors_;
    std::mt19937 &seed_;
    double *new_address_states_;
    double *new_address_buffer_;
    double target_val_;
    double rate_{kPROBABILITY / kTIMESTEP};
    std::bernoulli_distribution status_{rate_ * kTIMESTEP};
    int steps_back_{};
    int delayed_steps_{};

  public:
    Simulation1d(std::mt19937 &seed)
        : states(full_states.topRows(kN_BIRDS).data()), buffer(full_states.topRows(kN_BIRDS).data()),
          timer_states(timers.topRows(kN_BIRDS).data()), timer_buffer(timers.topRows(kN_BIRDS).data()), seed_(seed)
    {

        Eigen::Matrix<double, kN_BIRDS, 2> positions{generate_initial_state()};
        y_states = positions.col(1);
        full_states.setZero();
        timers.setZero();
        for (int i = 0; i < kBUFFER_CYCLES + 1; ++i)
            full_states.col(3 * i) = positions.col(0);

        // initial offset of maps
        double *new_address = states.data() + (kN_BIRDS * 3);
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 3>>(new_address, kN_BIRDS, 3);

        double *new_address_timers = timers.data() + (kN_BIRDS * 2);
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, kN_BIRDS, 2);
    }

    Eigen::Matrix<double, kN_BIRDS, 2> generate_initial_state()
    {
        std::uniform_real_distribution<double> starting_x{-kSTARTING_INTERVAL / 2, kSTARTING_INTERVAL / 2};
        std::uniform_real_distribution<double> starting_y{kHEIGHT * (0.5f - kBOUND_OFFSET),
                                                          kHEIGHT * (0.5f + kBOUND_OFFSET)};
        Eigen::Matrix<double, kN_BIRDS, 2> starting_states;
        starting_states.col(0) = starting_states.col(0).NullaryExpr([&]() { return starting_x(seed_); });
        starting_states.col(1) = starting_states.col(1).NullaryExpr([&]() { return starting_y(seed_); });
        return starting_states;
    }

    void nearest__idx(const Eigen::Ref<const Eigen::Vector<double, kN_BIRDS>> positions, int target_idx)
    {
        target_val_ = positions(target_idx);
        idx_.setLinSpaced(0, kN_BIRDS - 1);

        std::nth_element(idx_.begin(), idx_.begin() + kM - 1, idx_.end(), [&](int i, int j) {
            return std::abs(positions(i) - target_val_) < std::abs(positions(j) - target_val_);
        });

        for (int i = 0; i < kM; ++i)
        {
            neighbors_(i) = idx_(i);
        }
        std::sort(neighbors_.begin(), neighbors_.end());
    }

    void flock_acceleration(Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 3>> states)
    {
        // Eigen::internal::set_is_malloc_allowed(false);

        auto positions{states.col(0)};
        auto velocities{states.col(1)};
        auto status_{states.col(2)};
        double cm{kALI / kM};
        for (int i = 0; i < kN_BIRDS; ++i)
        {
            nearest__idx(positions, i);
            closest_neighbors_(i) = neighbors_(0);
            auto neighbor_positions{positions(neighbors_).array()};
            auto neighbor_velocities{velocities(neighbors_).array()};
            auto pos_diff{neighbor_positions - positions(i)};
            auto vel_diff{neighbor_velocities - velocities(i)};
            auto a_sum_1{((pos_diff.array()) / ((pos_diff).array().abs().square() + kEPSILON))};
            auto a_sum_2{(cm * (vel_diff) + kATT * (pos_diff))};
            double acceleration{-kREP * a_sum_1.sum() + (1.0 - status_(i)) * a_sum_2.sum()};
            acceleration_vec_(i) = acceleration;
        }
        packed_result_ << acceleration_vec_,
            closest_neighbors_; // packing closest neighbors for persistence distance tracking
        // Eigen::internal::set_is_malloc_allowed(true);
    }
    void shift_back() // shifts both maps, remapping is not a heap allocation!
    {
        steps_back_ = (steps_back_ + 1) % (kBUFFER_CYCLES + 1); // circular modulus
        new_address_states_ = full_states.data() + (steps_back_ * kN_BIRDS * 3);
        new (&states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 3>>(new_address_states_, kN_BIRDS, 3);

        double *new_address_timers = timers.data() + (steps_back_ * kN_BIRDS * 2);
        new (&timer_states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, kN_BIRDS, 2);

        delayed_steps_ = (steps_back_ + 1) % (kBUFFER_CYCLES + 1);
        new_address_buffer_ = full_states.data() + delayed_steps_ * kN_BIRDS * 3;
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 3>>(new_address_buffer_, kN_BIRDS, 3);

        double *new_address_timers_delay = timers.data() + delayed_steps_ * kN_BIRDS * 2;
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers_delay, kN_BIRDS, 2);
    }

    void update_state(Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 3>> states)
    {
        flock_acceleration(buffer);

        for (int i = 0; i < kN_BIRDS; ++i)
        {
            bool is_leader = states.col(2)[i] > 0.5;
            double dist_to_neighbor = std::abs(states.col(0)[i] - states.col(0)[packed_result_(kN_BIRDS + i)]);

            if (is_leader)
            {
                // LEADER LOGIC
                bool expired = timer_states.col(0)[i] <= 0;
                bool strayed = dist_to_neighbor > kPD;

                if (expired || strayed)
                {
                    // DEMOTE
                    buffer.col(2)[i] = 0;
                    timer_buffer.col(1)[i] = kRT;
                    timer_buffer.col(0)[i] = 0;
                }
                else
                {
                    // REMAIN LEADER
                    buffer.col(2)[i] = 1;
                    if (timer_states.col(0)[i] > 0)
                        timer_buffer.col(0)[i] = timer_states.col(0)[i] - kTIMESTEP;
                }
            }
            else
            {
                // FOLLOWER LOGIC
                if (timer_states.col(1)[i] > 0)
                {
                    // REFRACTORY PERIOD
                    timer_buffer.col(1)[i] = std::max(0.0, timer_states.col(1)[i] - kTIMESTEP);
                    buffer.col(2)[i] = 0;
                }
                else
                {
                    // ELIGIBLE FOR PROMOTION
                    double roll = status_(seed_);
                    buffer.col(2)[i] = roll;
                    if (roll > 0.5)
                    {
                        timer_buffer.col(0)[i] = kPT; // Assign persistence
                    }
                }
            }
        }
        buffer.col(1) = states.col(1) + packed_result_(Eigen::seq(0, kN_BIRDS - 1)) *
                                            kTIMESTEP; //"future" position, will become present after shift
        buffer.col(0) = states.col(0) + states.col(1) * kTIMESTEP;
    }
};

class Simulation2d
{
  public:
    Eigen::Matrix<double, kN_BIRDS, 4 * (1 + kBUFFER_CYCLES)> full_states;
    Eigen::Matrix<double, kN_BIRDS, 2 * (1 + kBUFFER_CYCLES)> timers;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 4>> states;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 4>> buffer;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>> timer_states;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>> timer_buffer;
    Eigen::Vector<double, kN_BIRDS> y_states;

  private:
    Eigen::Vector<double, kN_BIRDS> acceleration_vec_;
    Eigen::Vector<double, kN_BIRDS> closest_neighbors_;
    Eigen::Vector<double, 2 * kN_BIRDS> packed_result_;
    Eigen::Vector<std::int32_t, 2 * kN_BIRDS> left_idx_;
    Eigen::Vector<std::int32_t, 2 * kN_BIRDS> right_idx_;
    Eigen::Vector<float, 2 * kN_BIRDS> split_values_;
    Eigen::Vector<int, kN_BIRDS> tree_idx_;

    Eigen::Vector<std::uint8_t, 2 * kN_BIRDS> dimensions_;
    Eigen::Vector<int, kM> neighbors_;
    Eigen::Vector<std::uint8_t, kTREE_MEDIAN_SAMPLE> sample_indices_; // small sample of values for median

    struct Tree_task_
    {
        int pool_idx{};
        int start{};
        int end{};
        Tree_task_(int pool, int start, int end) : pool_idx{pool}, start{start}, end{end}
        {
        }
    };
    std::vector<Tree_task_> tree_stack_;

    std::mt19937 seed_;
    double rate_{kPROBABILITY / kTIMESTEP};
    std::bernoulli_distribution status_{rate_ * kTIMESTEP};
    double target_val_;
    double *new_address_states_;
    double *new_address_buffer_;
    int steps_back_{};
    int delayed_steps_{};
    int dim_{};
    int pool_ticker_{};

  public:
    Simulation2d(std::mt19937 &mt)
        : states(full_states.topRows(kN_BIRDS).data()), buffer(full_states.topRows(kN_BIRDS).data()),
          timer_states(timers.topRows(kN_BIRDS).data()), timer_buffer(timers.topRows(kN_BIRDS).data()), seed_(mt)
    {
        Eigen::Matrix<double, kN_BIRDS, 2> positions{generate_initial_state()};
        y_states = positions.col(1);
        full_states.setZero();
        timers.setZero();
        for (int i = 0; i < kBUFFER_CYCLES + 1; ++i)
            full_states.col(4 * i) = positions.col(0);

        // initial offset of maps
        double *new_address = states.data() + (kN_BIRDS * 4);
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 4>>(new_address, kN_BIRDS, 4);

        double *new_address_timers = timers.data() + (kN_BIRDS * 2);
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, kN_BIRDS, 2);

        tree_stack_.reserve(120); // rough estimate of max depth
    }

    Eigen::Matrix<double, kN_BIRDS, 2> generate_initial_state()
    {
        std::uniform_real_distribution<double> starting_x{-kSTARTING_INTERVAL / 2, kSTARTING_INTERVAL / 2};
        std::uniform_real_distribution<double> starting_y{kHEIGHT * (0.5f - kBOUND_OFFSET),
                                                          kHEIGHT * (0.5f + kBOUND_OFFSET)};
        Eigen::Matrix<double, kN_BIRDS, 2> starting_states;
        starting_states.col(0) = starting_states.col(0).NullaryExpr([&]() { return starting_x(seed_); });
        starting_states.col(1) = starting_states.col(1).NullaryExpr([&]() { return starting_y(seed_); });
        return starting_states;
    }

    /* left idx,right idx = 2*nbirds
    split values = 2*nbirds
    tree_idx = nbirds
    dimension = 2*nbirds
    sample indicies = kMEDIAN_TREE_SAMPLE            investigate size allocations

    pool idx
    start
    end
    */
    void construct_tree(const Eigen::Ref<const Eigen::Matrix<double, kN_BIRDS, 2>> &positions)
    {
        std::iota(sample_indices_.begin(), sample_indices_.end(),
                  0); // reset and initialize sample median proxy list, tree node indices
        std::iota(tree_idx_.begin(), tree_idx_.end(), 0); // tree_idx is position proxy sort mask
        pool_ticker_ = 0;                                 // set ticker (node counter) to first non-root node??
        tree_stack_.emplace_back(pool_ticker_, 0, kN_BIRDS);

        int count = 1;

        while (!tree_stack_.empty())
        {
            std::cout << "round: " << count << std::endl << "------" << std::endl;
            Tree_task_ Task = tree_stack_.back();
            tree_stack_.pop_back();

            int n_node_birds = Task.end - Task.start;
            if (n_node_birds <= kLEAF_SIZE)
            {                                           // we directly calculate remaining leaf cluster using SIMD
                left_idx_(Task.pool_idx) = -Task.start; // left/right_idx mark NODE indices
                right_idx_(Task.pool_idx) = -Task.end;  // here, position indices used for sentinel?
                continue;
            }
            // auto x_segment = positions.col(0).segment(Task.start, n_node_birds);
            // auto y_segment = positions.col(1).segment(Task.start, n_node_birds);

            auto x_segment = positions.col(0)(tree_idx_.segment(Task.start, n_node_birds)).eval();
            auto y_segment = positions.col(1)(tree_idx_.segment(Task.start, n_node_birds)).eval();

            dim_ = (std::abs(x_segment.maxCoeff() - x_segment.minCoeff()) > // PROFILE
                    std::abs(y_segment.maxCoeff() - y_segment.minCoeff()))
                       ? 0
                       : 1;
            dimensions_(Task.pool_idx) = dim_;

            int endpoint = std::min(kTREE_MEDIAN_SAMPLE, n_node_birds);
            std::nth_element(sample_indices_.begin(), sample_indices_.begin() + endpoint / 2,
                             sample_indices_.begin() +
                                 endpoint, // sample_indices proxy sort mask of SUBSET of positions
                             [&](int i, int j) {
                                 return positions(Task.start + i, dim_) < positions(Task.start + j, dim_);
                             }); // issue
            int sample_median_idx = (sample_indices_.size() % 2 == 1) ? sample_indices_(sample_indices_.size() / 2 + 1)
                                                                      : sample_indices_(sample_indices_.size() / 2);
            std::cout << sample_indices_ << std::endl << "------" << std::endl;
            std::cout << positions(sample_indices_, dim_) << std::endl << "------" << std::endl;
            auto tree_midpoint_pointer = std::partition(
                tree_idx_.begin() + Task.start,
                tree_idx_.begin() + Task.end, // partition task segment of position mask based on positions
                [&](int i) {
                    return positions(i, dim_) < positions(sample_median_idx, dim_);
                }); // indexes full positions using sample mask index
            std::cout << tree_idx_ << std::endl << "---------" << std::endl;
            int tree_midpoint_offset = std::distance(tree_idx_.begin() + Task.start, tree_midpoint_pointer);
            double split = positions(tree_midpoint_offset, dim_);

            split_values_(Task.pool_idx) = split;
            left_idx_(Task.pool_idx) = ++pool_ticker_; // assigning child NODES
            tree_stack_.emplace_back(pool_ticker_, Task.start, Task.start + tree_midpoint_offset);
            right_idx_(Task.pool_idx) = ++pool_ticker_;
            tree_stack_.emplace_back(pool_ticker_, Task.start + tree_midpoint_offset + 1, Task.end);
            ++count;
        }
    }

    // void flock_acceleration(Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 4>> states)
    // {
    //     // Eigen::internal::set_is_malloc_allowed(false);

    //     auto positions{states.col(0)};
    //     auto velocities{states.col(1)};
    //     auto status_{states.col(2)};
    //     double cm{kALI / kM};
    //     for (int i = 0; i < kN_BIRDS; ++i)
    //     {
    //         kd_nearest_neighbors(positions, i);
    //         closest_neighbors_(i) = neighbors_(0);
    //         auto neighbor_positions{positions(neighbors_).array()};
    //         auto neighbor_velocities{velocities(neighbors_).array()};
    //         auto pos_diff{neighbor_positions - positions(i)};
    //         auto vel_diff{neighbor_velocities - velocities(i)};
    //         auto a_sum_1{((pos_diff.array()) / ((pos_diff).array().abs().square() + kEPSILON))};
    //         auto a_sum_2{(cm * (vel_diff) + kATT * (pos_diff))};
    //         double acceleration{-kREP * a_sum_1.sum() + (1.0 - status_(i)) * a_sum_2.sum()};
    //         acceleration_vec_(i) = acceleration;
    //     }
    //     packed_result_ << acceleration_vec_,
    //         closest_neighbors_; // packing closest neighbors for persistence distance tracking
    //     // Eigen::internal::set_is_malloc_allowed(true);
    // }

    void shift_back() // shifts both maps, remapping is not a heap allocation!
    {
        steps_back_ = (steps_back_ + 1) % (kBUFFER_CYCLES + 1); // circular modulus
        new_address_states_ = full_states.data() + (steps_back_ * kN_BIRDS * 4);
        new (&states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 4>>(new_address_states_, kN_BIRDS, 4);

        double *new_address_timers = timers.data() + (steps_back_ * kN_BIRDS * 2);
        new (&timer_states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, kN_BIRDS, 2);

        delayed_steps_ = (steps_back_ + 1) % (kBUFFER_CYCLES + 1);
        new_address_buffer_ = full_states.data() + delayed_steps_ * kN_BIRDS * 4;
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 4>>(new_address_buffer_, kN_BIRDS, 4);

        double *new_address_timers_delay = timers.data() + delayed_steps_ * kN_BIRDS * 2;
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers_delay, kN_BIRDS, 2);
    }

    // void update_state(Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 4>> states)
    // {
    //     flock_acceleration(buffer);

    //     for (int i = 0; i < kN_BIRDS; ++i)
    //     {
    //         bool is_leader = states.col(2)[i] > 0.5;
    //         double dist_to_neighbor = std::abs(states.col(0)[i] - states.col(0)[packed_result_(kN_BIRDS + i)]);

    //         if (is_leader)
    //         {
    //             // LEADER LOGIC
    //             bool expired = timer_states.col(0)[i] <= 0;
    //             bool strayed = dist_to_neighbor > kPD;

    //             if (expired || strayed)
    //             {
    //                 // DEMOTE
    //                 buffer.col(2)[i] = 0;
    //                 timer_buffer.col(1)[i] = kRT;
    //                 timer_buffer.col(0)[i] = 0;
    //             }
    //             else
    //             {
    //                 // REMAIN LEADER
    //                 buffer.col(2)[i] = 1;
    //                 if (timer_states.col(0)[i] > 0)
    //                     timer_buffer.col(0)[i] = timer_states.col(0)[i] - kTIMESTEP;
    //             }
    //         }
    //         else
    //         {
    //             // FOLLOWER LOGIC
    //             if (timer_states.col(1)[i] > 0)
    //             {
    //                 // REFRACTORY PERIOD
    //                 timer_buffer.col(1)[i] = std::max(0.0, timer_states.col(1)[i] - kTIMESTEP);
    //                 buffer.col(2)[i] = 0;
    //             }
    //             else
    //             {
    //                 // ELIGIBLE FOR PROMOTION
    //                 double roll = status_(seed_);
    //                 buffer.col(2)[i] = roll;
    //                 if (roll > 0.5)
    //                 {
    //                     timer_buffer.col(0)[i] = kPT; // Assign persistence
    //                 }
    //             }
    //         }
    //     }
    //     buffer.col(1) = states.col(1) + packed_result_(Eigen::seq(0, kN_BIRDS - 1)) *
    //                                         kTIMESTEP; //"future" position, will become present after shift
    //     buffer.col(0) = states.col(0) + states.col(1) * kTIMESTEP;
    // }
};