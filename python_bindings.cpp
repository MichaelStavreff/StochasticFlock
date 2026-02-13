#include "simulation.hpp"
#include "visual.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using Sim1d = Simulation1d<Eigen::Dynamic>;
using Sim2d = Simulation2d<Eigen::Dynamic>;
void init_simulation(Sim2d &sim);

void bind_parameters(py::module &m)
{
    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("kN_BIRDS", &Parameters::kN_BIRDS)
        .def_readwrite("kM", &Parameters::kM)
        .def_readwrite("kDELAY", &Parameters::kDELAY)
        .def_readwrite("kPT", &Parameters::kPT)
        .def_readwrite("kPD", &Parameters::kPD)
        .def_readwrite("kRT", &Parameters::kRT)
        .def_readwrite("kREP", &Parameters::kREP)
        .def_readwrite("kALI", &Parameters::kALI)
        .def_readwrite("kATT", &Parameters::kATT)
        .def_readwrite("kTIMESTEP", &Parameters::kTIMESTEP)
        .def_readwrite("kROUNDS", &Parameters::kROUNDS)
        .def_readwrite("kPROBABILITY", &Parameters::kPROBABILITY)
        .def_readwrite("kBOX_SIZE", &Parameters::kBOX_SIZE)
        .def_readwrite("kEPSILON", &Parameters::kEPSILON)
        .def_readwrite("kBUFFER_CYCLES", &Parameters::kBUFFER_CYCLES); // must be calculated manually in python
}

PYBIND11_MODULE(stochastic_flock, m)
{
    m.doc() = "Unified Bird Flocking Simulation (1D & 2D)";

    bind_parameters(m);

    py::class_<std::mt19937>(m, "MT19937").def(py::init<uint32_t>());

    py::class_<Sim1d>(m, "Simulation1d")
        .def(py::init<Parameters &, std::mt19937 &>())
        .def("update",
             [](Sim1d &self) {
                 self.update_state();
                 self.shift_back();
             })
        .def_property_readonly("states", [](Sim1d &self) { return self.states; })
        .def_property_readonly("positions", [](Sim1d &self) { return self.states.col(0); })
        .def_property_readonly("velocities", [](Sim1d &self) { return self.states.col(1); })
        .def_property_readonly("status", [](Sim1d &self) { return self.states.col(2); })
        .def_property_readonly("timers", [](Sim1d &self) { return self.timer_states; });

    py::class_<Sim2d>(m, "Simulation2d")
        .def(py::init<Parameters &, std::mt19937 &, const std::optional<Eigen::Matrix<double, Eigen::Dynamic, 4>> &>(),
             py::arg("params"), py::arg("seed"), py::arg("start_conditions") = std::nullopt)
        .def("update",
             [](Sim2d &self) {
                 self.update_state();
                 self.shift_back();
             })
        .def("debug_tree", &Sim2d::debug_print_tree, py::arg("i") = 0, py::arg("indent") = 0)
        .def("show",
             [](Sim2d &self) {
                 py::gil_scoped_release release; // Allows other Python threads to run while window is open
                 init_simulation_fontless(self);
             })

        .def_property_readonly("states", [](Sim2d &self) { return self.states; })
        .def_property_readonly("positions", [](Sim2d &self) { return self.states.leftCols(2); })
        .def_property_readonly("velocities", [](Sim2d &self) { return self.states.middleCols(2, 2); })
        .def_property_readonly("status", [](Sim2d &self) { return self.states.col(4); })
        .def_property_readonly("timers", [](Sim2d &self) { return self.timer_states; });
}