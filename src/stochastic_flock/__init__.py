"""
Stochastic Flock
================
A high-performance agent-based simulation of bird flocking based on the
"All-Leader" model by Cristiani et al. (2021).

The model uses second-order delayed stochastic differential equations where
leadership is a transient, self-organized state rather than a fixed hierarchy.

Classes:
    Parameters:     Configuration container for physical and social forces.
    Simulation2d:   The 2D core simulation engine.
    MT19937:        Seedable Mersenne Twister random number generator.
"""

from . import stochastic_flock_core as _core


class Parameters(_core.Parameters):
    """
    Configuration container for the flocking model.

    This class defines the physical forces, social behaviors, and numerical
    integration constants used by the Simulation2d engine.

    ### Core Attributes
    * **kN_BIRDS** (int): Total number of agents in the simulation.
    * **kM** (float): Number of nearest topological neighbors each bird interacts with.
    * **kDELAY** (int): Reaction time delay (δ) expressed in number of timesteps.

    ### Leadership Logic
    * **kPROBABILITY** (float): Probability per step that a follower becomes a leader.
    * **kPT** (float): Persistence Time; max duration an agent stays in the leader state.
    * **kPD** (float): Persistence Distance; max distance from neighbors before losing leadership.
    * **kRT** (float): Refractory Time; cooldown period before an agent can lead again.

    ### Social Forces
    * **kREP** (float): Repulsion coefficient (prevents collisions).
    * **kALI** (float): Alignment coefficient (velocity matching).
    * **kATT** (float): Attraction coefficient (flock cohesion).
    * **kEPSILON** (float): Small constant to prevent division by zero in force math.

    ### Simulation Setup
    * **kTIMESTEP** (float): The integration timestep (dt).
    * **kROUNDS** (int): Total number of simulation steps to execute.
    * **kBOX_SIZE** (float): Initial side-length of the random distribution area.
    * **kBUFFER_CYCLES** (int): Size of the history buffer.
      *Note: Must satisfy `kBUFFER_CYCLES * kTIMESTEP >= kDELAY`.*
    """

    kN_BIRDS: int
    kM: float  # Number of topological neighbors
    kDELAY: int  # Time delay (delta)
    kPT: float  # Persistence Time (p)
    kPD: float  # Persistence Distance (d)
    kRT: float  # Refractory Time (tau)
    kREP: float  # Repulsion coefficient
    kALI: float  # Alignment coefficient
    kATT: float  # Attraction coefficient
    kTIMESTEP: float  # dt
    kROUNDS: int
    kPROBABILITY: float  # P(follower -> leader)
    kBOX_SIZE: float
    kEPSILON: float
    kBUFFER_CYCLES: int  # Must satisfy kBUFFER_CYCLES * kTIMESTEP >= kDELAY

    def to_dict(self) -> dict:
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("k") and not callable(getattr(self, attr))
        }

    def __repr__(self) -> str:
        header = f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>"
        params = (
            f"Parameters(N={self.kN_BIRDS}, M={self.kM}, dt={self.kTIMESTEP}, "
            f"prob={self.kPROBABILITY}, delay={self.kDELAY})"
        )

        return f"{header}\n{params}"


Simulation2d = _core.Simulation2d
MT19937 = _core.MT19937


__all__ = ["Parameters", "Simulation2d", "MT19937"]
