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
struct Parameters
{
    int kN_BIRDS{200};
    int kM{7};                    // bird set size,        7
    double kDELAY{0.1};           // delay,                0.1
    double kPT{200};              // persistence time,     700
    double kPD{100};              // persistence distance squared, 20^2
    double kRT{800};              // refractory time,      800
    double kREP{2.5};             // repulsion force,      2.5
    double kALI{3};               // alignment force,      3
    double kATT{0.01};            // attraction force,     0.01
    double kTIMESTEP{0.05};       //                       0.1
    double kPROBABILITY{0.00002}; //                       0.0002
    double kEPSILON{0.001};       // std::numeric_limits<double>::epsilon() for machine epsilon, 0.0001
    int kBUFFER_CYCLES{static_cast<int>(kDELAY / kTIMESTEP)};

    int kFRAMERATE{60};
    int kROUNDS{10000};
    // Higher dimensional parameters
    int kTREE_MEDIAN_SAMPLE{30}; // could profile
    int kLEAF_SIZE{16};          // double once using floats to fit SIMD registers better, could profile
    double kPRUNE_EPS{0.01};     // for approximate KD tree pruning
    double kAPPROX_PRUNE_FACTOR{1.0 / ((1.0 + kPRUNE_EPS) * (1.0 + kPRUNE_EPS))}; // Precompute this!
                                                                                  // 1D window
    float kWIDTH{2200};
    float kHEIGHT{400};
    float kBOUND_OFFSET{0.1};    // purely visual "height" of 1D line
    int kSTARTING_INTERVAL{200}; // bound which birds are initialized
                                 // 2D window
    float kWIDTH_2D{1200};
    float kHEIGHT_2D{900};
    float kBOX_SIZE{200}; // starting interval in 2D
};
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

    void nearest_idx(const Eigen::Ref<const Eigen::Vector<double, kN_BIRDS>> &positions, int target_idx)
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
            nearest_idx(positions, i);
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
    Eigen::Matrix<double, kN_BIRDS, 5 * (1 + kBUFFER_CYCLES)> full_states;
    Eigen::Matrix<double, kN_BIRDS, 2 * (1 + kBUFFER_CYCLES)> timers;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>> states;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>> buffer;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>> timer_states;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>> timer_buffer;
    Eigen::Matrix<double, kN_BIRDS, 2> proxy_positions_;
    Eigen::Vector<double, kN_BIRDS> y_states;
    Eigen::Vector<int, kN_BIRDS> tree_idx_;               // move back to private once finished
    Eigen::Vector<std::int32_t, 2 * kN_BIRDS> left_idx_;  // move back to private once finished
    Eigen::Vector<std::int32_t, 2 * kN_BIRDS> right_idx_; // move back to private once finished
    std::array<int, kN_BIRDS> traversal_stack_;           // move back to private once finished
    Eigen::Vector<int, kN_BIRDS> inverse_idx_;

  private:
    Eigen::Vector<double, kN_BIRDS> acceleration_vec_x;
    Eigen::Vector<double, kN_BIRDS> acceleration_vec_y;
    Eigen::Vector<double, kN_BIRDS> closest_neighbors_;
    Eigen::Vector<double, kN_BIRDS> bird_dimensions_positions_unroll_;
    Eigen::Vector<double, 3 * kN_BIRDS> packed_result_;
    Eigen::Vector<double, 2 * kN_BIRDS> split_values_;

    Eigen::Vector<std::uint8_t, 2 * kN_BIRDS> dimensions_;
    Eigen::Vector<double, kLEAF_SIZE> squared_distances_;
    Eigen::Vector<int, kM> neighbors_;
    Eigen::Vector<double, kM> persistent_neighbors_dist_;
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
    double backtrace_sqr_radius_;
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
        {
            full_states.col(5 * i) = positions.col(0);
            full_states.col(5 * i + 1) = positions.col(1);
        }
        // initial offset of maps
        double *new_address = states.data() + (kN_BIRDS * 5);
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>>(new_address, kN_BIRDS, 5);

        double *new_address_timers = timers.data() + (kN_BIRDS * 2);
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, kN_BIRDS, 2);

        tree_stack_.reserve(120); // rough estimate of max depth
    }

    Eigen::Matrix<double, kN_BIRDS, 2> generate_initial_state()
    {
        // Use kBOX_SIZE to keep the birds in the blue square you drew
        std::uniform_real_distribution<double> dist{-kBOX_SIZE / 2.0, kBOX_SIZE / 2.0};

        Eigen::Matrix<double, kN_BIRDS, 2> starting_states;
        // Center both X and Y around 0.0
        starting_states.col(0) = starting_states.col(0).NullaryExpr([&]() { return dist(seed_); });
        starting_states.col(1) = starting_states.col(1).NullaryExpr([&]() { return dist(seed_); });

        return starting_states;
    }
    void debug_print_tree(int i = 0, int indent = 0)
    {
        if (i == -1)
            return; // Empty node

        // Print indentation
        for (int j = 0; j < indent; ++j)
            std::cout << "  ";

        if (left_idx_(i) < 0)
        {
            // This is a Leaf
            int task_pos = -left_idx_(i) - 1; // undo offset
            int task_n = -right_idx_(i);
            std::cout << "[LEAF] pos: " << task_pos << " count: " << task_n << " (IDs: ";
            for (int k = 0; k < task_n; ++k)
            {
                std::cout << tree_idx_(task_pos + k) << (k == task_n - 1 ? "" : ", ");
            }
            std::cout << ")" << std::endl;
        }
        else
        {
            // This is an Internal Node
            std::cout << "[NODE " << i << "] Split Dim: " << (int)dimensions_(i) << " Value: " << split_values_(i)
                      << std::endl;

            // Recurse
            debug_print_tree(left_idx_(i), indent + 1);
            debug_print_tree(right_idx_(i), indent + 1);
        }
    }

    void construct_tree(const Eigen::Ref<const Eigen::Matrix<double, kN_BIRDS, 2>> &positions)
    {
        std::iota(sample_indices_.begin(), sample_indices_.end(), 0);
        std::iota(tree_idx_.begin(), tree_idx_.end(), 0);
        pool_ticker_ = 0;
        tree_stack_.emplace_back(pool_ticker_, 0, kN_BIRDS);

        int count = 1;

        while (!tree_stack_.empty())
        {
            Tree_task_ Task = tree_stack_.back();
            tree_stack_.pop_back();

            int n_node_birds = Task.end - Task.start;
            if (n_node_birds <= kLEAF_SIZE)
            { // we directly calculate remaining leaf cluster using SIMD
                left_idx_(Task.pool_idx) =
                    -(Task.start +
                      1); // -1 offset to handle -0=0 case, negative values are sentinels for traversal algorithm
                right_idx_(Task.pool_idx) = -n_node_birds;
                ++count;
                continue;
            }

            auto x_segment =
                positions.col(0)(tree_idx_.segment(Task.start, n_node_birds))
                    .eval(); // eval incurrs hidden temporaries overhead but avoids risking lazy evaluation bugs
            auto y_segment = positions.col(1)(tree_idx_.segment(Task.start, n_node_birds)).eval();
            dim_ = (std::abs(x_segment.maxCoeff() - x_segment.minCoeff()) > // could profile
                    std::abs(y_segment.maxCoeff() - y_segment.minCoeff()))
                       ? 0
                       : 1;
            dimensions_(Task.pool_idx) = dim_;

            int endpoint = std::min(kTREE_MEDIAN_SAMPLE, n_node_birds);
            std::nth_element(tree_idx_.begin() + Task.start, tree_idx_.begin() + Task.start + endpoint / 2,
                             tree_idx_.begin() + Task.start + endpoint,
                             [&](int i, int j) { return positions(i, dim_) < positions(j, dim_); });
            int sample_median_idx = *(tree_idx_.begin() + Task.start + endpoint / 2);

            auto tree_midpoint_pointer =
                std::partition(tree_idx_.begin() + Task.start, tree_idx_.begin() + Task.end,
                               [&](int i) { return positions(i, dim_) < positions(sample_median_idx, dim_); });
            int tree_midpoint_offset = std::distance(tree_idx_.begin() + Task.start, tree_midpoint_pointer);
            if (tree_midpoint_offset == 0) // preventing empty splits
                ++tree_midpoint_offset;
            else if (tree_midpoint_offset == n_node_birds)
                --tree_midpoint_offset;

            double split = positions(sample_median_idx, dim_);
            split_values_(Task.pool_idx) = split;
            left_idx_(Task.pool_idx) = ++pool_ticker_;
            tree_stack_.emplace_back(pool_ticker_, Task.start, Task.start + tree_midpoint_offset);
            right_idx_(Task.pool_idx) = ++pool_ticker_;
            tree_stack_.emplace_back(pool_ticker_, Task.start + tree_midpoint_offset, Task.end);
            ++count;
            if (pool_ticker_ > dimensions_.size())
                std::cout << "ticker too big: " << pool_ticker_ << std::endl;
        }
        // populate proxy position matrix for traversal
        for (int k{}; auto bird : tree_idx_)
        {
            proxy_positions_.col(0)(k) = positions.col(0)(bird);
            proxy_positions_.col(1)(k) = positions.col(1)(bird);
            ++k;
        }
        // inverse lookup for bird_idx in traversal
        for (int i = 0; i < kN_BIRDS; ++i)
            inverse_idx_[tree_idx_[i]] = i;
        // debug_print_tree();
    }
    void traverse_backtrace(const Eigen::Ref<const Eigen::Matrix<double, kN_BIRDS, 2>> &positions, int pos_bird_idx)
    {
        int proxy_bird_idx = inverse_idx_(pos_bird_idx);
        double target_x = proxy_positions_(proxy_bird_idx, 0);
        double target_y = proxy_positions_(proxy_bird_idx, 1);

        neighbors_.setConstant(-1);
        persistent_neighbors_dist_.setConstant(std::numeric_limits<double>::max());
        backtrace_sqr_radius_ = std::numeric_limits<double>::max();
        int neighbors_found = 0;

        int i = 0;
        int stack_counter = 0;

        while (true)
        {
            while (left_idx_(i) >= 0)
            {
                traversal_stack_[stack_counter++] = i;
                int dim = (int)dimensions_(i);
                double val = (dim == 0) ? target_x : target_y;

                if (val < split_values_(i))
                    i = left_idx_(i);
                else
                    i = right_idx_(i);
            }

            int task_pos = (-left_idx_(i)) - 1; // Decode sentinel to get starting position in proxy matrix
            int task_n = -right_idx_(i);

            squared_distances_.head(task_n) =
                (proxy_positions_.col(0).segment(task_pos, task_n).array() - target_x).square() +
                (proxy_positions_.col(1).segment(task_pos, task_n).array() - target_y).square();

            for (int j = 0; j < task_n; ++j)
            {
                double d2 = squared_distances_(j);

                // Check if bird qualifies for the top K (kM) neighbors
                if (neighbors_found < kM || d2 < backtrace_sqr_radius_)
                {
                    int global_proxy_idx = task_pos + j;

                    int curr = (neighbors_found < kM) ? neighbors_found : (kM - 1);

                    if (neighbors_found == kM && d2 >= persistent_neighbors_dist_(kM - 1))
                        continue;

                    while (curr > 0 && d2 < persistent_neighbors_dist_(curr - 1))
                    {
                        persistent_neighbors_dist_(curr) = persistent_neighbors_dist_(curr - 1);
                        neighbors_(curr) = neighbors_(curr - 1);
                        --curr;
                    }

                    persistent_neighbors_dist_(curr) = d2;
                    neighbors_(curr) = global_proxy_idx;

                    if (neighbors_found < kM)
                        neighbors_found++;

                    backtrace_sqr_radius_ = persistent_neighbors_dist_(neighbors_found - 1);
                }
            }

            bool found_sibling = false;
            while (stack_counter > 0)
            {
                int parent = traversal_stack_[--stack_counter];
                int dim = (int)dimensions_(parent);
                double val = (dim == 0) ? target_x : target_y;

                double diff = val - split_values_(parent);
                double diff2 = diff * diff;

                if (neighbors_found < kM || diff2 < backtrace_sqr_radius_ / kAPPROX_PRUNE_FACTOR)
                {
                    i = (val < split_values_(parent)) ? right_idx_(parent) : left_idx_(parent);
                    found_sibling = true;
                    break;
                }
            }

            if (!found_sibling)
                break;
        }
    }
    void flock_acceleration(Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 5>> states)
    {
        // Eigen::internal::set_is_malloc_allowed(false); // Safety check for heap allocations

        auto positions = states.leftCols<2>();
        auto velocities = states.middleCols<2>(2);
        auto status_ = states.col(4);
        double cm = kALI / kM;

        Eigen::Matrix<double, kN_BIRDS, 2> acc_matrix;

        for (int i = 0; i < kN_BIRDS; ++i)
        {
            traverse_backtrace(positions, i);
            closest_neighbors_(i) = tree_idx_(neighbors_(0));

            auto n_pos = positions(tree_idx_(neighbors_), Eigen::all).array();
            auto n_vel = velocities(tree_idx_(neighbors_), Eigen::all).array();

            auto pos_diff = n_pos.rowwise() - positions.row(i).array();
            auto vel_diff = n_vel.rowwise() - velocities.row(i).array();

            auto dist_sq = pos_diff.square().rowwise().sum();

            auto a_sum_1 = pos_diff.colwise() / (dist_sq + kEPSILON);

            auto a_sum_2 = (cm * vel_diff) + (kATT * pos_diff);

            acc_matrix.row(i) = -kREP * a_sum_1.colwise().sum() + (1.0 - status_(i)) * a_sum_2.colwise().sum();
        }

        packed_result_.segment(0, kN_BIRDS) = acc_matrix.col(0);
        packed_result_.segment(kN_BIRDS, kN_BIRDS) = acc_matrix.col(1);
        packed_result_.segment(2 * kN_BIRDS, kN_BIRDS) = closest_neighbors_;

        // Eigen::internal::set_is_malloc_allowed(true);
    }

    void shift_back() // shifts both maps, remapping is not a heap allocation!
    {
        steps_back_ = (steps_back_ + 1) % (kBUFFER_CYCLES + 1); // circular modulus
        new_address_states_ = full_states.data() + (steps_back_ * kN_BIRDS * 5);
        new (&states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>>(new_address_states_, kN_BIRDS, 5);

        double *new_address_timers = timers.data() + (steps_back_ * kN_BIRDS * 2);
        new (&timer_states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, kN_BIRDS, 2);

        delayed_steps_ = (steps_back_ + 1) % (kBUFFER_CYCLES + 1);
        new_address_buffer_ = full_states.data() + delayed_steps_ * kN_BIRDS * 5;
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>>(new_address_buffer_, kN_BIRDS, 5);

        double *new_address_timers_delay = timers.data() + delayed_steps_ * kN_BIRDS * 2;
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers_delay, kN_BIRDS, 2);
    }

    void update_state(Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 5>> states)
    {
        construct_tree(buffer.leftCols(2));
        flock_acceleration(buffer);

        for (int i = 0; i < kN_BIRDS; ++i)
        {
            bool is_leader = states.col(4)[i] > 0.5;

            int neighbor_id = static_cast<int>(packed_result_(2 * kN_BIRDS + i));

            double dx = states.col(0)[i] - states.col(0)[neighbor_id];
            double dy = states.col(1)[i] - states.col(1)[neighbor_id];
            double dist_to_neighbor = dx * dx + dy * dy;

            if (is_leader)
            {
                // LEADER LOGIC
                bool expired = timer_states.col(0)[i] <= 0;
                bool strayed = dist_to_neighbor > kPD;

                if (expired || strayed)
                {
                    // DEMOTE
                    buffer.col(4)[i] = 0; // Update status in Column 4
                    timer_buffer.col(1)[i] = kRT;
                    timer_buffer.col(0)[i] = 0;
                }
                else
                {
                    // REMAIN LEADER
                    buffer.col(4)[i] = 1;
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
                    buffer.col(4)[i] = 0;
                }
                else
                {
                    // ELIGIBLE FOR PROMOTION
                    double roll = status_(seed_);
                    buffer.col(4)[i] = roll;
                    if (roll > 0.5)
                    {
                        timer_buffer.col(0)[i] = kPT; // Assign persistence
                    }
                }
            }
        }

        buffer.col(2) = states.col(2) + packed_result_.segment(0, kN_BIRDS) * kTIMESTEP;
        buffer.col(3) = states.col(3) + packed_result_.segment(kN_BIRDS, kN_BIRDS) * kTIMESTEP;

        buffer.col(0) = states.col(0) + states.col(2) * kTIMESTEP;
        buffer.col(1) = states.col(1) + states.col(3) * kTIMESTEP;
    }
};