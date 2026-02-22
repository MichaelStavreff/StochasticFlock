#include "pch_python.hpp" //included outside Cmake just for static checker
#include "simulation.hpp"

using Sim2d = Simulation2d<Eigen::Dynamic>;

// void bind_parameters(py::module &m)
// {
//     py::class_<Parameters>(m, "Parameters")
//         .def(py::init<>())
//         .def_readwrite("kN_BIRDS", &Parameters::kN_BIRDS)
//         .def_readwrite("kM", &Parameters::kM)
//         .def_readwrite("kDELAY", &Parameters::kDELAY)
//         .def_readwrite("kPT", &Parameters::kPT)
//         .def_readwrite("kPD", &Parameters::kPD)
//         .def_readwrite("kRT", &Parameters::kRT)
//         .def_readwrite("kREP", &Parameters::kREP)
//         .def_readwrite("kALI", &Parameters::kALI)
//         .def_readwrite("kATT", &Parameters::kATT)
//         .def_readwrite("kTIMESTEP", &Parameters::kTIMESTEP)
//         .def_readwrite("kROUNDS", &Parameters::kROUNDS)
//         .def_readwrite("kPROBABILITY", &Parameters::kPROBABILITY)
//         .def_readwrite("kBOX_SIZE", &Parameters::kBOX_SIZE)
//         .def_readwrite("kEPSILON", &Parameters::kEPSILON)
//         .def_readwrite("kBUFFER_CYCLES", &Parameters::kBUFFER_CYCLES); // must be calculated manually in python
// }

// PYBIND11_MODULE(stochastic_flock, m)
// {
//     m.doc() = "Agent-Based Bird Flocking Simulation";

//     bind_parameters(m);

//     py::class_<std::mt19937>(m, "MT19937").def(py::init<uint32_t>());

//     py::class_<Sim2d>(m, "Simulation2d")
//         .def(py::init<Parameters &, std::mt19937 &, const std::optional<Eigen::Matrix<double, Eigen::Dynamic, 4>>
//         &>(),
//              py::arg("params"), py::arg("seed"), py::arg("start_conditions") = std::nullopt)
//         .def("update",
//              [](Sim2d &self) {
//                  self.update_state();
//                  self.shift_back();
//              })
//         .def("debug_tree", &Sim2d::debug_print_tree, py::arg("i") = 0, py::arg("indent") = 0)

//         .def_property_readonly("states", [](Sim2d &self) { return self.states; })
//         .def_property_readonly("positions", [](Sim2d &self) { return self.states.leftCols(2); })
//         .def_property_readonly("velocities", [](Sim2d &self) { return self.states.middleCols(2, 2); })
//         .def_property_readonly("status", [](Sim2d &self) { return self.states.col(4); })
//         .def_property_readonly("timers", [](Sim2d &self) { return self.timer_states; });
// }

void bind_parameters(py::module &m)
{
    py::class_<Parameters>(m, "Parameters", "Container for simulation constants and social force coefficients.")
        .def(py::init<>())
        .def_readwrite("kN_BIRDS", &Parameters::kN_BIRDS, "Total number of agents (birds) in the simulation.")
        .def_readwrite("kM", &Parameters::kM, "Number of nearest topological neighbors each agent interacts with.")
        .def_readwrite("kDELAY", &Parameters::kDELAY, "Reaction time delay (delta) in the differential equations.")
        .def_readwrite("kPT", &Parameters::kPT, "Persistence Time: Max time an agent remains a leader.")
        .def_readwrite("kPD", &Parameters::kPD,
                       "Persistence Distance: Max distance from neighbors before losing leadership.")
        .def_readwrite("kRT", &Parameters::kRT,
                       "Refractory Time: Cooldown period before an agent can become a leader again.")
        .def_readwrite("kREP", &Parameters::kREP, "Repulsion force coefficient (collision avoidance).")
        .def_readwrite("kALI", &Parameters::kALI, "Alignment force coefficient (velocity matching).")
        .def_readwrite("kATT", &Parameters::kATT, "Attraction force coefficient (flock cohesion).")
        .def_readwrite("kTIMESTEP", &Parameters::kTIMESTEP, "Integration timestep (dt).")
        .def_readwrite("kROUNDS", &Parameters::kROUNDS, "Number of simulation steps to run.")
        .def_readwrite("kPROBABILITY", &Parameters::kPROBABILITY,
                       "Probability per step for a follower to trigger a turn (become leader).")
        .def_readwrite("kBOX_SIZE", &Parameters::kBOX_SIZE, "Size of the initial distribution area.")
        .def_readwrite("kEPSILON", &Parameters::kEPSILON,
                       "Small constant to prevent division by zero in force calculations.")
        .def_readwrite("kBUFFER_CYCLES", &Parameters::kBUFFER_CYCLES,
                       "History buffer size required to handle the time delay.");
}

PYBIND11_MODULE(stochastic_flock_core, m)
{
    m.doc() = "C++ backend for the All-Leader Bird Flocking Simulation.";

    bind_parameters(m);

    py::class_<std::mt19937>(m, "MT19937", "Mersenne Twister 19937 pseudo-random generator.")
        .def(py::init<uint32_t>(), py::arg("seed"), "Initialize the generator with a deterministic seed.");

    py::class_<Sim2d>(m, "Simulation2d", "The 2D simulation engine managing agent states and physics.")
        .def(py::init<Parameters &, std::mt19937 &, const std::optional<Eigen::Matrix<double, Eigen::Dynamic, 4>> &>(),
             py::arg("params"), py::arg("seed"), py::arg("start_conditions") = std::nullopt,
             "Constructs the simulation. If start_conditions is None, agents are distributed randomly.")
        .def(
            "update",
            [](Sim2d &self) {
                self.update_state();
                self.shift_back();
            },
            "Calculates forces, updates velocities/positions, and handles leader transitions for one timestep.")
        .def("debug_tree", &Sim2d::debug_print_tree, py::arg("i") = 0, py::arg("indent") = 0,
             "Prints a text representation of the internal KD-Tree for spatial neighbor lookups.")

        .def_property_readonly(
            "states", [](Sim2d &self) { return self.states; },
            "Full (N, 5) state matrix: [X, Y, VX, VY, Status], read-only.")
        .def_property_readonly(
            "positions", [](Sim2d &self) { return self.states.leftCols(2); },
            "View of (N, 2) agent coordinates, read-only.")
        .def_property_readonly(
            "velocities", [](Sim2d &self) { return self.states.middleCols(2, 2); },
            "Read-only view of (N, 2) velocity vectors.")
        .def_property_readonly(
            "status", [](Sim2d &self) { return self.states.col(4); },
            "Agent roles: 0.0 for Follower, 1.0 for Leader, read-only.")
        .def_property_readonly(
            "timers", [](Sim2d &self) { return self.timer_states; },
            "Internal countdowns for leadership duration and refractory periods for followers, read-only.");
}