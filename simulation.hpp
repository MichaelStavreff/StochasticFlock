#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream> //endl
#include <random>

#include "constants.hpp"

//  revisit nearest neighbors for 1D, pointer walk better?
// cmake
// evaluate where to include floats or doubles (during calculations/cast back to float for storage)
// verify SIMD/cache misses
struct Parameters
{
    int kN_BIRDS{200};
    int kM{7};                   // bird set size,        7
    double kDELAY{0.1};          // delay,                0.1
    double kPT{700};             // persistence time,     700
    double kPD{400};             // persistence distance squared, 20^2
    double kRT{800};             // refractory time,      800
    double kREP{2.5};            // repulsion force,      2.5
    double kALI{3};              // alignment force,      3
    double kATT{0.01};           // attraction force,     0.01
    double kTIMESTEP{0.1};       //                       0.1
    double kPROBABILITY{0.0002}; //                       0.0002
    double kEPSILON{0.001};      // std::numeric_limits<double>::epsilon() for machine epsilon, 0.0001
    int kBUFFER_CYCLES{static_cast<int>(kDELAY / kTIMESTEP)};

    int kFRAMERATE{60};
    int kROUNDS{100000};
    // Higher dimensional parameters
    int kTREE_MEDIAN_SAMPLE{30}; // could profile
    int kLEAF_SIZE{16};          // double once using floats to fit SIMD registers better, could profile
    double kPRUNE_EPS{0.01};     // for approximate KD tree pruning
    double kAPPROX_PRUNE_FACTOR{1.0 / ((1.0 + kPRUNE_EPS) * (1.0 + kPRUNE_EPS))};
    // 1D window
    float kWIDTH{2200};
    float kHEIGHT{400};
    float kBOUND_OFFSET{0.1};    // purely visual "height" of 1D line
    int kSTARTING_INTERVAL{200}; // bound which birds are initialized
    // 2D window
    float kWIDTH_2D{2000};
    float kHEIGHT_2D{1200};
    float kBOX_SIZE{200}; // starting interval in 2D
};

template <int kN_BIRDS = Eigen::Dynamic> class Simulation1d
{
  public:
    Parameters p;
    Eigen::Matrix<double, kN_BIRDS, Eigen::Dynamic> full_states;
    Eigen::Matrix<double, kN_BIRDS, Eigen::Dynamic> timers;

  private:
    Eigen::Vector<double, (kN_BIRDS == Eigen::Dynamic ? Eigen::Dynamic : 2 * kN_BIRDS)> packed_result_;
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
    Eigen::Vector<int, Eigen::Dynamic> neighbors_;
    std::mt19937 &seed_;
    double *new_address_states_;
    double *new_address_buffer_;
    double target_val_;
    double rate_{p.kPROBABILITY / p.kTIMESTEP};
    std::bernoulli_distribution status_{rate_ * p.kTIMESTEP};
    int steps_back_{};
    int delayed_steps_{};

  public:
    Simulation1d(Parameters &params, std::mt19937 &mt)
        : p(params), seed_(mt),
          full_states(get_actual_n(params), 3 * (static_cast<int>(params.kDELAY / params.kTIMESTEP) + 1)),
          timers(get_actual_n(params), 2 * (static_cast<int>(params.kDELAY / params.kTIMESTEP) + 1)),
          states(full_states.data(), get_actual_n(params), 3), timer_states(timers.data(), get_actual_n(params), 2),
          buffer(states), timer_buffer(timer_states)
    {

        const int n = states.rows();
        const int m = params.kM;

        Eigen::Matrix<double, kN_BIRDS, 2> positions = generate_initial_state();
        idx_.resize(n);
        neighbors_.resize(m);
        acceleration_vec_.resize(n);
        closest_neighbors_.resize(n);
        packed_result_.resize(2 * n);

        y_states = positions.col(1);
        full_states.setZero();
        timers.setZero();
        for (int i = 0; i < params.kBUFFER_CYCLES + 1; ++i)
            full_states.col(3 * i) = positions.col(0);

        // initial offset of maps
        double *new_address = states.data() + (n * 3);
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 3>>(new_address, n, 3);

        double *new_address_timers = timers.data() + (n * 2);
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, n, 2);
    }

    static int get_actual_n(const Parameters &params)
    {
        if constexpr (kN_BIRDS == Eigen::Dynamic)
            return params.kN_BIRDS;
        else
            return kN_BIRDS;
    }

    Eigen::Matrix<double, kN_BIRDS, 2> generate_initial_state()
    {
        const int n = states.rows();
        std::uniform_real_distribution<double> starting_x{-p.kSTARTING_INTERVAL / 2.0, p.kSTARTING_INTERVAL / 2.0};
        std::uniform_real_distribution<double> starting_y{p.kHEIGHT * (0.5f - p.kBOUND_OFFSET),
                                                          p.kHEIGHT * (0.5f + p.kBOUND_OFFSET)};
        Eigen::Matrix<double, kN_BIRDS, 2> starting_states(n, 2);
        for (int i = 0; i < n; ++i)
        {
            starting_states(i, 0) = starting_x(seed_);
            starting_states(i, 1) = starting_y(seed_);
        }

        return starting_states;
    }

    void nearest_idx(const Eigen::Ref<const Eigen::Vector<double, kN_BIRDS>> &positions, int target_idx)
    {
        const int n = states.rows();
        target_val_ = positions(target_idx);
        idx_.setLinSpaced(n, 0, n - 1);

        int search_count = std::min(n - 1, p.kM);

        std::nth_element(idx_.begin(), idx_.begin() + search_count, idx_.end(), [&](int i, int j) {
            return std::abs(positions(i) - target_val_) < std::abs(positions(j) - target_val_);
        });

        int filled = 0;
        for (int i = 0; i <= search_count; ++i)
        {
            if (idx_(i) == target_idx)
                continue;
            if (filled < p.kM)
            {
                neighbors_(filled++) = idx_(i);
            }
        }

        if (filled > 0)
        {
            auto closest_it = std::min_element(neighbors_.begin(), neighbors_.begin() + filled, [&](int i, int j) {
                return std::abs(positions(i) - target_val_) < std::abs(positions(j) - target_val_);
            });
            std::iter_swap(neighbors_.begin(), closest_it);
        }
    }
    void flock_acceleration(Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 3>> past_states,
                            Eigen::Ref<Eigen::Vector<double, kN_BIRDS>> current_status)
    {
        // Eigen::internal::set_is_malloc_allowed(false);

        const int n = past_states.rows();

        auto positions{past_states.col(0)};
        auto velocities{past_states.col(1)};
        double cm{p.kALI / p.kM};
        for (int i = 0; i < n; ++i)
        {
            nearest_idx(positions, i);
            closest_neighbors_(i) = neighbors_(0);
            auto neighbor_positions{positions(neighbors_).array()};
            auto neighbor_velocities{velocities(neighbors_).array()};
            auto pos_diff{neighbor_positions - positions(i)};
            auto vel_diff{neighbor_velocities - velocities(i)};
            auto a_sum_1{((pos_diff.array()) / ((pos_diff).array().abs().square() + p.kEPSILON))};
            auto a_sum_2{(cm * (vel_diff) + p.kATT * (pos_diff))};
            double acceleration{-p.kREP * a_sum_1.sum() + (1.0 - current_status(i)) * a_sum_2.sum()};
            acceleration_vec_(i) = acceleration;
        }
        packed_result_ << acceleration_vec_,
            closest_neighbors_; // packing closest neighbors for persistence distance tracking
        // Eigen::internal::set_is_malloc_allowed(true);
    }
    void shift_back() // shifts both maps, remapping is not a heap allocation!
    {
        const int n = states.rows();
        steps_back_ = (steps_back_ + 1) % (p.kBUFFER_CYCLES + 1); // circular modulus
        new_address_states_ = full_states.data() + (steps_back_ * n * 3);
        new (&states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 3>>(new_address_states_, n, 3);

        double *new_address_timers = timers.data() + (steps_back_ * n * 2);
        new (&timer_states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, n, 2);

        delayed_steps_ = (steps_back_ + 1) % (p.kBUFFER_CYCLES + 1);
        new_address_buffer_ = full_states.data() + delayed_steps_ * n * 3;
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 3>>(new_address_buffer_, n, 3);

        double *new_address_timers_delay = timers.data() + delayed_steps_ * n * 2;
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers_delay, n, 2);
    }

    void update_state()
    {
        const int n = states.rows();
        flock_acceleration(buffer, states.col(2));

        for (int i = 0; i < n; ++i)
        {
            bool is_leader = states.col(2)[i] > 0.5;
            double dist_to_neighbor = std::abs(states.col(0)[i] - states.col(0)[packed_result_(n + i)]);

            if (is_leader)
            {
                // LEADER LOGIC
                bool expired = timer_states.col(0)[i] <= 0;
                bool strayed = dist_to_neighbor * dist_to_neighbor > p.kPD;

                if (expired || strayed)
                {
                    // DEMOTE
                    buffer.col(2)[i] = 0;
                    timer_buffer.col(1)[i] = p.kRT;
                    timer_buffer.col(0)[i] = 0;
                }
                else
                {
                    // REMAIN LEADER
                    buffer.col(2)[i] = 1;
                    if (timer_states.col(0)[i] > 0)
                        timer_buffer.col(0)[i] = timer_states.col(0)[i] - p.kTIMESTEP;
                }
            }
            else
            {
                // FOLLOWER LOGIC
                if (timer_states.col(1)[i] > 0)
                {
                    // REFRACTORY PERIOD
                    timer_buffer.col(1)[i] = std::max(0.0, timer_states.col(1)[i] - p.kTIMESTEP);
                    buffer.col(2)[i] = 0;
                }
                else
                {
                    // ELIGIBLE FOR PROMOTION
                    double roll = status_(seed_);
                    buffer.col(2)[i] = roll;
                    if (roll > 0.5)
                    {
                        timer_buffer.col(0)[i] = p.kPT; // Assign persistence
                    }
                }
            }
        }
        buffer.col(1) = states.col(1) + packed_result_(Eigen::seq(0, n - 1)) *
                                            p.kTIMESTEP; //"future" position, will become present after shift
        buffer.col(0) = states.col(0) + states.col(1) * p.kTIMESTEP;
    }
};

template <int kN_BIRDS = Eigen::Dynamic> class Simulation2d
{
  public:
    Parameters p;
    Eigen::Matrix<double, kN_BIRDS, Eigen::Dynamic> full_states;
    Eigen::Matrix<double, kN_BIRDS, Eigen::Dynamic> timers;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>> states;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>> buffer;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>> timer_states;
    Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>> timer_buffer;
    Eigen::Matrix<double, kN_BIRDS, 2> proxy_positions_;
    Eigen::Matrix<double, kN_BIRDS, 2> acc_matrix;
    Eigen::Vector<double, kN_BIRDS> y_states;
    Eigen::Vector<int, kN_BIRDS> tree_idx_; // move back to private once finished
    Eigen::Vector<std::int32_t, (kN_BIRDS == Eigen::Dynamic ? Eigen::Dynamic : 2 * kN_BIRDS)>
        left_idx_; // move back to private once finished
    Eigen::Vector<std::int32_t, (kN_BIRDS == Eigen::Dynamic ? Eigen::Dynamic : 2 * kN_BIRDS)>
        right_idx_;                                // move back to private once finished
    Eigen::Vector<int, kN_BIRDS> traversal_stack_; // move back to private once finished
    Eigen::Vector<int, kN_BIRDS> inverse_idx_;

  private:
    Eigen::Vector<double, kN_BIRDS> acceleration_vec_x;
    Eigen::Vector<double, kN_BIRDS> acceleration_vec_y;
    Eigen::Vector<double, kN_BIRDS> closest_neighbors_;
    Eigen::Vector<double, kN_BIRDS> bird_dimensions_positions_unroll_;
    Eigen::Vector<double, (kN_BIRDS == Eigen::Dynamic ? Eigen::Dynamic : 3 * kN_BIRDS)> packed_result_;
    Eigen::Vector<double, (kN_BIRDS == Eigen::Dynamic ? Eigen::Dynamic : 2 * kN_BIRDS)> split_values_;

    Eigen::Vector<std::uint8_t, (kN_BIRDS == Eigen::Dynamic ? Eigen::Dynamic : 2 * kN_BIRDS)> dimensions_;
    Eigen::Vector<double, kLEAF_SIZE> squared_distances_;
    Eigen::Vector<int, Eigen::Dynamic> neighbors_;
    Eigen::Vector<double, Eigen::Dynamic> persistent_neighbors_dist_;
    Eigen::Vector<std::uint8_t, kTREE_MEDIAN_SAMPLE> sample_indices_; // small sample of values for median
    Eigen::Vector<double, kN_BIRDS> scratchpad_x;
    Eigen::Vector<double, kN_BIRDS> scratchpad_y;

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
    double rate_{p.kPROBABILITY / p.kTIMESTEP};
    std::bernoulli_distribution status_{rate_ * p.kTIMESTEP};
    double target_val_;
    double backtrace_sqr_radius_;
    double *new_address_states_;
    double *new_address_buffer_;
    int steps_back_{};
    int delayed_steps_{};
    int dim_{};
    int pool_ticker_{};

  public:
    Simulation2d(Parameters &param, std::mt19937 &mt,
                 const std::optional<Eigen::Matrix<double, kN_BIRDS, 4>> &initial_conditions = std::nullopt)
        : p(param), seed_(mt),
          full_states(get_actual_n(param), 5 * (static_cast<int>(param.kDELAY / param.kTIMESTEP) + 1)),
          timers(get_actual_n(param), 2 * (static_cast<int>(param.kDELAY / param.kTIMESTEP) + 1)),
          states(full_states.data(), get_actual_n(param), 5), timer_states(timers.data(), get_actual_n(param), 2),
          buffer(states), timer_buffer(timer_states)
    {
        const int n = states.rows();
        const int m = param.kM;

        Eigen::Matrix<double, kN_BIRDS, 2> positions(n, 2);
        Eigen::Matrix<double, kN_BIRDS, 2> velocities(n, 2);

        if (!initial_conditions.has_value())
        {
            positions = generate_initial_state();
            velocities.setZero();
        }
        else
        {
            if ((*initial_conditions).rows() != n)
            {
                throw std::runtime_error("Initial conditions row count must match kN_BIRDS");
            }
            positions = (*initial_conditions).leftCols(2);
            velocities = (*initial_conditions).rightCols(2);
        }
        proxy_positions_.resize(n, 2);
        y_states.resize(n);
        tree_idx_.resize(n);
        left_idx_.resize(2 * n);
        right_idx_.resize(2 * n);
        traversal_stack_.resize(n);
        inverse_idx_.resize(n);
        acceleration_vec_x.resize(n);
        acceleration_vec_y.resize(n);
        closest_neighbors_.resize(n);
        bird_dimensions_positions_unroll_.resize(n);
        packed_result_.resize(3 * n);
        split_values_.resize(2 * n);
        dimensions_.resize(2 * n);
        neighbors_.resize(m);
        persistent_neighbors_dist_.resize(m);
        acc_matrix.resize(n, 2);
        scratchpad_x.resize(n);
        scratchpad_y.resize(n);

        y_states = positions.col(1);
        full_states.setZero();
        timers.setZero();
        for (int i = 0; i < p.kBUFFER_CYCLES + 1; ++i)
        {
            full_states.col(5 * i) = positions.col(0);
            full_states.col(5 * i + 1) = positions.col(1);
            full_states.col(5 * i + 2) = velocities.col(0); // velocities
            full_states.col(5 * i + 3) = velocities.col(1);
        }
        // initial offset of maps
        double *new_address = states.data() + (n * 5);
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>>(new_address, n, 5);

        double *new_address_timers = timers.data() + (n * 2);
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, n, 2);

        tree_stack_.reserve(120); // rough estimate of max depth
    }

    static int get_actual_n(const Parameters &params)
    {
        if constexpr (kN_BIRDS == Eigen::Dynamic)
            return params.kN_BIRDS;
        else
            return kN_BIRDS;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 2> generate_initial_state()
    {
        const int n = states.rows();
        std::uniform_real_distribution<double> dist{-p.kBOX_SIZE / 2.0, p.kBOX_SIZE / 2.0};

        Eigen::Matrix<double, kN_BIRDS, 2> starting_states(n, 2);
        for (int i = 0; i < n; ++i)
        {
            starting_states(i, 0) = dist(seed_);
            starting_states(i, 1) = dist(seed_);
        }
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
        const int n = states.rows();
        std::iota(sample_indices_.begin(), sample_indices_.end(), 0);
        std::iota(tree_idx_.begin(), tree_idx_.end(), 0);
        pool_ticker_ = 0;
        tree_stack_.emplace_back(pool_ticker_, 0, n);

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

            dimensions_(Task.pool_idx) = dim_;

            scratchpad_x.head(n_node_birds) = positions.col(0)(tree_idx_.segment(Task.start, n_node_birds));
            scratchpad_y.head(n_node_birds) = positions.col(1)(tree_idx_.segment(Task.start, n_node_birds));

            auto x_view = scratchpad_x.head(n_node_birds);
            auto y_view = scratchpad_y.head(n_node_birds);
            dim_ = (std::abs(x_view.maxCoeff() - x_view.minCoeff()) > std::abs(y_view.maxCoeff() - y_view.minCoeff()))
                       ? 0
                       : 1;

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
        for (int i = 0; i < n; ++i)
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
                int global_proxy_idx = task_pos + j;
                int real_bird_id = tree_idx_(global_proxy_idx);

                if (real_bird_id == pos_bird_idx)
                    continue;

                double d2 = squared_distances_(j);

                // Check if bird qualifies for the top K (kM) neighbors
                if (neighbors_found < p.kM || d2 < backtrace_sqr_radius_)
                {
                    int curr = (neighbors_found < p.kM) ? neighbors_found : (p.kM - 1);

                    if (neighbors_found == p.kM && d2 >= persistent_neighbors_dist_(p.kM - 1))
                        continue;

                    while (curr > 0 && d2 < persistent_neighbors_dist_(curr - 1))
                    {
                        persistent_neighbors_dist_(curr) = persistent_neighbors_dist_(curr - 1);
                        neighbors_(curr) = neighbors_(curr - 1);
                        --curr;
                    }

                    persistent_neighbors_dist_(curr) = d2;
                    neighbors_(curr) = global_proxy_idx; // Storing the proxy index

                    if (neighbors_found < p.kM)
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

                if (neighbors_found < p.kM || diff2 < backtrace_sqr_radius_ / kAPPROX_PRUNE_FACTOR)
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
    void flock_acceleration(const Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 5>> &past_states,
                            const Eigen::Ref<const Eigen::Vector<double, kN_BIRDS>> &current_status)
    {
        // Eigen::internal::set_is_malloc_allowed(false); // Safety check for heap allocations
        const int n = past_states.rows();

        auto positions = past_states.leftCols(2);
        auto velocities = past_states.middleCols(2, 2);
        double cm = p.kALI / p.kM;

        for (int i = 0; i < n; ++i)
        {
            traverse_backtrace(positions, i);
            closest_neighbors_(i) = tree_idx_(neighbors_(0));

            auto n_pos = positions(tree_idx_(neighbors_), Eigen::all).array();
            auto n_vel = velocities(tree_idx_(neighbors_), Eigen::all).array();

            auto pos_diff = n_pos.rowwise() - positions.row(i).array();
            auto vel_diff = n_vel.rowwise() - velocities.row(i).array();

            auto dist_sq = pos_diff.square().rowwise().sum();

            auto a_sum_1 = pos_diff.colwise() / (dist_sq + p.kEPSILON);

            auto a_sum_2 = (cm * vel_diff) + (p.kATT * pos_diff);

            acc_matrix.row(i) = -p.kREP * a_sum_1.colwise().sum() + (1.0 - current_status(i)) * a_sum_2.colwise().sum();
        }

        packed_result_.segment(0, n) = acc_matrix.col(0);
        packed_result_.segment(n, n) = acc_matrix.col(1);
        packed_result_.segment(2 * n, n) = closest_neighbors_;

        // Eigen::internal::set_is_malloc_allowed(true);
    }

    void shift_back() // shifts both maps, remapping is not a heap allocation!
    {
        const int n = states.rows();
        steps_back_ = (steps_back_ + 1) % (p.kBUFFER_CYCLES + 1); // circular modulus
        new_address_states_ = full_states.data() + (steps_back_ * n * 5);
        new (&states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>>(new_address_states_, n, 5);

        double *new_address_timers = timers.data() + (steps_back_ * n * 2);
        new (&timer_states) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers, n, 2);

        delayed_steps_ = (steps_back_ + 1) % (p.kBUFFER_CYCLES + 1);
        new_address_buffer_ = full_states.data() + delayed_steps_ * n * 5;
        new (&buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 5>>(new_address_buffer_, n, 5);

        double *new_address_timers_delay = timers.data() + delayed_steps_ * n * 2;
        new (&timer_buffer) Eigen::Map<Eigen::Matrix<double, kN_BIRDS, 2>>(new_address_timers_delay, n, 2);
    }

    void update_state()
    {
        const int n = states.rows();
        construct_tree(buffer.leftCols(2));
        flock_acceleration(buffer, states.col(4));

        for (int i = 0; i < n; ++i)
        {
            bool is_leader = states.col(4)[i] > 0.5;

            int neighbor_id = static_cast<int>(packed_result_(2 * n + i));

            double dx = states.col(0)[i] - states.col(0)[neighbor_id];
            double dy = states.col(1)[i] - states.col(1)[neighbor_id];
            double dist_to_neighbor = dx * dx + dy * dy;

            if (is_leader)
            {
                // LEADER LOGIC
                bool expired = timer_states.col(0)[i] <= 0;
                bool strayed = dist_to_neighbor > p.kPD;

                if (expired || strayed)
                {
                    // DEMOTE
                    buffer.col(4)[i] = 0; // Update status in Column 4
                    timer_buffer.col(1)[i] = p.kRT;
                    timer_buffer.col(0)[i] = 0;
                }
                else
                {
                    // REMAIN LEADER
                    buffer.col(4)[i] = 1;
                    if (timer_states.col(0)[i] > 0)
                        timer_buffer.col(0)[i] = timer_states.col(0)[i] - p.kTIMESTEP;
                }
            }
            else
            {
                // FOLLOWER LOGIC
                if (timer_states.col(1)[i] > 0)
                {
                    // REFRACTORY PERIOD
                    timer_buffer.col(1)[i] = std::max(0.0, timer_states.col(1)[i] - p.kTIMESTEP);
                    buffer.col(4)[i] = 0;
                }
                else
                {
                    // ELIGIBLE FOR PROMOTION
                    double roll = status_(seed_);
                    buffer.col(4)[i] = roll;
                    if (roll > 0.5)
                    {
                        timer_buffer.col(0)[i] = p.kPT; // Assign persistence
                    }
                }
            }
        }

        buffer.col(2) = states.col(2) + packed_result_.segment(0, n) * p.kTIMESTEP;
        buffer.col(3) = states.col(3) + packed_result_.segment(n, n) * p.kTIMESTEP;

        buffer.col(0) = states.col(0) + states.col(2) * p.kTIMESTEP;
        buffer.col(1) = states.col(1) + states.col(3) * p.kTIMESTEP;
    }
};