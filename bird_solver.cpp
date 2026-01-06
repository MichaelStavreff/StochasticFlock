
#include "constants.hpp"
#include "simulation.hpp"
using namespace Eigen;
using namespace std;
struct Simulation;

void init_simulation(Simulation &simulation, const int simulation_rounds);

int main()
{
    int simulation_rounds{1000};
    Simulation Sim;
    // state matrix (cross sectional, all birds), delay matrix, matrix vector views
    // all followers at t=0
    // skip first delay amount of time to fill buffer

    init_simulation(Sim, simulation_rounds);
    return 0;
}
