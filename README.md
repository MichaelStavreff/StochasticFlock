

# High Performance C++ Flocking Simulation for Stochastic Modelling
> *Michael Stavreff,*
> *December 17, 2025*
## Development & Compilation

This project uses **CMake** and a **Makefile** to manage the C++ build process, dependencies (Eigen, pybind11), and performance optimizations.

### Prerequisites

To build the project from source, you will need:
* **C++20 Compiler** (GCC 10+ or Clang 11+)
* **CMake** (3.18+)
* **Python 3.8+** (with `python3-venv` and `python3-dev`)
* **SFML 2.5** (Optional: only required for the visual standalone solver)
  ```bash 
  sudo apt install libsfml-dev
  ```
### Python Usage

If you simply wish to use the simulation in a Python environment without modifying the C++ source:

```Bash
pip install stochastic_flock
```
```Python
>>> import stochastic_flock
params = stochastic_flock.Parameters()
seed = stochastic_flock.MT19937(30)
>>> sim = stochastic_flock.Simulation2d(params, seed)
```
### Developer Setup
1. Clone and create a virtual environment:

```Bash
git clone https://github.com/Mstavreff/stochastic_flock.git
cd stochastic_flock
python3 -m venv .venv
source .venv/bin/activate
```
2. Install build dependencies:
```Bash
pip install -r requirements.txt
make init
```

### Building the Project
The provided Makefile contains shortcuts for common build scenarios.
1. Standard Build (Python + C++ Solver)
Compiles the Python module into the root directory and the standalone solver into build/.

```Bash
make all
```
2. Native Hardware Optimization
Compiles using -march=native and -ffast-math. This produces the fastest possible binary for your specific CPU but is not portable to other hardware.

```Bash
make native
```
3. Profile-Guided Optimization (PGO) + Native 
For maximum performance, use PGO to optimize code paths based on simulation data for executable file only.
```Bash
make pgo
```
4. Debug Mode
Compiles with debug symbols (-g) and Undefined Behavior Sanitizers for use with GDB/LLDB.

```Bash
make debug
```
For local Python development, after running make, you can import the module directly inside the project directory.
## Introduction
Financial markets have famously exhibited flocking or herding behavior, most famously during crises and impending crashes. Such movements typically are completely unpredictable; the aim of this paper is to explore whether these movements are compltely unable to be modelled or are instead the product of some highly non-linear behavior requiring a novel approach. Naturally, birds in large flocks are an interesting candidate to model such emergent behavior in financial markets. Large flocks exhibit features which must be re-interpreted to a financial context, particularly in attraction, repulsion, turning behavior, and in the context of the particular paper, leader/follower dynamics:

### Reference Paper Overview
*The original paper is given here: https://arxiv.org/abs/2010.01990.* While the paper is short and concise and absolutely worth a read, some important details will be repeated. 

Cristiani et al.'s paper puts forward a second-order, delayed, stochastic differential equation to describe accelerations of bird agents. The stochastic element stems from birds having an exponential process (geometric in approximation) describing a follower -> leader transition where any attractive forces are dropped in their acceleration calculation and causing only repulsive-force trajectories. Additionally, birds react to the positions of others in a rank-ordered system of the nearest M birds, irregardless of distance; they also react to the delayed positions rather than the most instant information, something which creates more heavy movements and drifting of flocks in practice. These behaviors underline various interpretations worth discussing in the paper, mostly regarding trader behaviors or modelling portfolios of correlated equities in a sector which may be driven by such "social" forces and similarly experience shocks or new information in the form of a leader bird. 
