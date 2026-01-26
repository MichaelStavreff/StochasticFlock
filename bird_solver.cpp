
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

    std::mt19937 seed;
    seed.seed(std::random_device{}());

    Simulation2d Sim(seed);
    Eigen::Matrix<double, kN_BIRDS, 2> my_positions;
    my_positions.col(0) = Eigen::VectorXd::LinSpaced(kN_BIRDS, -50, 50);
    my_positions.col(1) = Eigen::VectorXd::LinSpaced(kN_BIRDS, -50, 50);
    std::cout << my_positions << std::endl;
    Sim.construct_tree(my_positions);
    // init_simulation(Sim);
}
