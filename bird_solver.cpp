
#include "constants.hpp"
#include "simulation.hpp"
using namespace Eigen;
using namespace std;
struct Simulation;

void init_simulation(Simulation &simulation);

int main()
{
    Simulation Sim(10000, 1);
    // state matrix (cross sectional, all birds), delay matrix, matrix vector views
    // all followers at t=0
    // skip first delay amount of time to fill buffer

    // for (int i{}; i < 10; ++i)
    // {
    //     cout << "Round" << " " << i << '\n';
    //     cout << Sim.full_states << '\n' << "-------------------------------------------" << '\n';
    //     Sim.update_state(Sim.states);
    // }

    init_simulation(Sim);
    return 0;
}
