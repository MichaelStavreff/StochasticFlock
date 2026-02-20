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
    int n_birds = 500;
    double att = 0.03;
    double rep = 2.5;
    std::cout << "birds: ";
    std::cin >> n_birds;
    std::cout << "\n";
    std::cout << "attraction: ";
    std::cin >> att;
    std::cout << "\n";
    std::cout << "repulsion: ";
    std::cin >> rep;
    std::cout << "\n";

    p.kWIDTH_2D = 2100;
    p.kHEIGHT_2D = 1300;
    p.kREP = rep;
    p.kATT = att;
    p.kFRAMERATE = 1000;
    p.kN_BIRDS = n_birds;
    p.kROUNDS = 1000000;
    std::mt19937 mt;
    mt.seed(std::random_device{}());
    // mt.seed(50);

    Eigen::Matrix<double, Eigen::Dynamic, 4> positions;
    positions.resize(p.kN_BIRDS, 4);
    positions = positions.setRandom() * 100;
    std::cout << positions;

    Simulation2d<> Sim(p, mt, positions);
    init_simulation(Sim);
}
