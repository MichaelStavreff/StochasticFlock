
#include "constants.hpp"
#include "simulation.hpp"

#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <format>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class Simulation2d;
// void init_simulation(Simulation1d &simulation);

int main()
{

    std::mt19937 mt;
    // mt.seed(std::random_device{}());
    mt.seed(44);
    std::uniform_real_distribution<> dist(-50, 50);

    Simulation2d Sim(mt);
    Eigen::Matrix<double, kN_BIRDS, 2> my_positions;
    // my_positions.col(0) = Eigen::VectorXd::LinSpaced(kN_BIRDS, -50, 50);
    // my_positions.col(1) = Eigen::VectorXd::LinSpaced(kN_BIRDS, -50, 50);
    my_positions.col(0) = Eigen::Vector<double, kN_BIRDS>::NullaryExpr(kN_BIRDS, [&]() { return dist(mt); });
    my_positions.col(1) = Eigen::Vector<double, kN_BIRDS>::NullaryExpr(kN_BIRDS, [&]() { return dist(mt); });
    std::cout << my_positions << std::endl;
    Sim.construct_tree(my_positions);

    Sim.run_integrity_check(100, my_positions, mt);
    // init_simulation(Sim);
}
