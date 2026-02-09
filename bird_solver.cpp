
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
struct Parameters;
void init_simulation(Simulation2d &simulation);

int main()
{
    Parameters p;
    std::mt19937 mt;
    // mt.seed(std::random_device{}());
    mt.seed(44);
    std::uniform_real_distribution<> dist(-50, 50);

    Simulation2d Sim(mt);
    init_simulation(Sim);
}
