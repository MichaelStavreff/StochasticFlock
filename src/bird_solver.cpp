#include "constants.hpp"
#include "simulation.hpp"
#include "visual.hpp"

#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <format>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main()
{

    Parameters p;
    p.kWIDTH_2D = 2000;
    p.kHEIGHT_2D = 1200;
    p.kREP = 2.5;
    p.kFRAMERATE = 1000;
    p.kN_BIRDS = 200;
    std::mt19937 mt;
    // mt.seed(std::random_device{}());
    mt.seed(50);

    Eigen::Matrix<double, 200, 4> positions;
    positions = positions.setRandom() * 100;
    std::cout << positions;

    Simulation2d<> Sim(p, mt, positions);
    init_simulation(Sim);
}
