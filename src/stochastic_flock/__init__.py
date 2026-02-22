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
