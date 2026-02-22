#include "constants.hpp"
#include "simulation.hpp"
#include "visual.hpp"

#include <random>
#include <stdexcept>
#include <string>

template <typename T> T get_parameter(T param, std::string name)
{
    std::string input;

    std::cout << "Enter " << name << " (or press Enter for default " << param << "): ";
    std::getline(std::cin, input);

    if (!input.empty())
    {
        try
        {
            return std::stoi(input);
        }
        catch (...)
        {
            std::cout << "Invalid input, keeping default.\n";
            return param;
        }
    }
    return param;
}

int main()
{

    Parameters p;

    p.kWIDTH_2D = 2100;
    p.kHEIGHT_2D = 1300;
    p.kN_BIRDS = get_parameter(300, "Bird count");
    p.kREP = get_parameter(2.5, "Repulsion force");
    p.kATT = get_parameter(0.01, "Attraction force");
    p.kM = get_parameter(7, "M-nearest birds");
    p.kPD = get_parameter(400, "Squared persistence distance");
    std::mt19937 mt;
    mt.seed(get_parameter(std::random_device{}(), "Seed value"));

    Eigen::Matrix<double, Eigen::Dynamic, 4> positions;
    positions.resize(p.kN_BIRDS, 4);
    positions = positions.setRandom() * 200;

    Simulation2d<> Sim(p, mt, positions);
    init_simulation(Sim);
}
