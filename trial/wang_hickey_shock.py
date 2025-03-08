#!/usr/bin/env python3

"""
Demo: Wang & Hickey (2020) Normal Shock for Non-Ideal Gas (Peng–Robinson EOS)

HOW TO USE:
1) Install requirements: pip install numpy scipy
2) Run: python wang_hickey_shock.py
3) Adjust the 'USER INPUT' section to match the upstream conditions from
   Wang & Hickey (2020) or any other reference data.
4) Compare the (p2, rho2, T2) solution to the reference, verifying correctness.
"""

import numpy as np
from math import sqrt
from scipy.optimize import fsolve

# ------------------------------------------------------------------------
# (A) Peng–Robinson EOS Implementation
# ------------------------------------------------------------------------
class PengRobinsonEOS:
    """
    Simplistic Peng–Robinson EOS for single-component CO2 demonstration.

    pressure(rho, T) -> returns p in Pa
    enthalpy(rho, T) -> returns h in J/kg (approximate)
    """

    def __init__(self):
        # Approximate constants for CO2
        self.Rg = 188.92        # J/(kg.K)  (Gas constant for CO2, ~8.314/ M_co2(44.01))
        self.Tc = 304.1282      # K
        self.Pc = 7.3773e6      # Pa
        self.omega = 0.225      # accentric factor for CO2

        # PR EOS parameters
        # a = 0.45724 * (R*Tc)^2 / Pc
        # b = 0.07780 * (R*Tc) / Pc
        self.a = 0.45724 * (self.Rg * self.Tc)**2 / self.Pc
        self.b = 0.07780 * (self.Rg * self.Tc) / self.Pc

        # kappa = 0.37464 + 1.54226*omega - 0.26992*omega^2
        self.kappa = 0.37464 + 1.54226*self.omega - 0.26992*(self.omega**2)

        # We'll do a simple constant-cp approach for demonstration
        # (In reality, you'd do a full enthalpy/residual approach)
        self.cp_ideal = 1200.0  # J/(kg.K), an approximate average for CO2

    def alpha(self, T):
        """ alpha(T) in Peng–Robinson """
        Tr = T / self.Tc
        return (1 + self.kappa*(1 - sqrt(Tr)))**2

    def pressure(self, rho, T):
        """
        p(rho, T) in Pa
        We interpret 'rho' in kg/m^3, so we need to find v (m^3/kg).
        """
        v = 1.0 / rho
        a_alpha = self.a * self.alpha(T)
        # PR formula:
        # p = R*T/(v - b) - a_alpha / (v^2 + 2*b*v - b^2)
        part1 = self.Rg*T / (v - self.b)
        part2 = a_alpha / (v**2 + 2*self.b*v - self.b**2)
        p = part1 - part2
        return p

    def enthalpy(self, rho, T):
        """
        Approx enthalpy in J/kg, h = e + p/rho.
        We'll do: h ~ cp_ideal * T, ignoring strong real-gas corrections
        for demonstration. (Better to do partial integrals of PR EOS.)
        """
        return self.cp_ideal * T

# ------------------------------------------------------------------------
# (B) Wang & Hickey Shock Functions (Continuity-Momentum, Continuity-Energy)
# ------------------------------------------------------------------------
def shock_residual(vars_, eos, rho1, u1, h1, p1):
    """
    Given (rho2, T2), return (CM, CE) = (0, 0) for the normal shock.

    CM = p2 - p1 - rho1*u1^2 (1 - rho2/rho1)
    CE = h2 - h1 - 0.5*u1^2 (1 - (rho2^2)/(rho1^2))
    with the expansions from Wang & Hickey (2020),
    except we store them in a more typical form:

        CM = p2 - p1 - (rho1 * u1^2) * (1 - rho2/rho1)
        CE = (h2 - h1) - 0.5 * u1^2 [1 - (rho2^2 / rho1^2)]

    We'll do h = e + p/rho, but here we approximate h2 = cp_ideal*T2.
    """
    rho2, T2 = vars_
    p2 = eos.pressure(rho2, T2)
    h2 = eos.enthalpy(rho2, T2)  # approx
    # continuity => rho1 * u1 = rho2 * u2
    # => u2 = (rho1 u1)/rho2
    u2 = (rho1*u1) / rho2

    # momentum form from W&H is: p1 + rho1*u1^2 = p2 + rho2*u2^2
    # => rearr => p2 - p1 = rho1*u1^2 - rho2*u2^2
    # => p2 - p1 = rho1*u1^2 - rho2( (rho1*u1)^2 / rho2^2 ) = ...
    # which matches a "shock function" approach. We'll do a simpler approach:
    # We'll define:
    CM = (p2 - p1) - (rho1*u1**2)*(1 - rho2/rho1)
    # energy form: h1 + 0.5 u1^2 = h2 + 0.5 u2^2
    # => h2 - h1 = 0.5 (u1^2 - u2^2)
    # => rearr => h2 - h1 = 0.5 [u1^2 - (rho1*u1/rho2)^2 ]
    # => ...
    # We'll define:
    CE = (h2 - h1) - 0.5*(u1**2)*(1 - (rho2/rho1)**2)

    return (CM, CE)

# ------------------------------------------------------------------------
# (C) Demo function to solve for normal shock
# ------------------------------------------------------------------------
def solve_normal_shock(eos, rho1, T1, u1):
    """
    Solve for (rho2, T2, p2, u2) using the shock approach of Wang & Hickey (2020).

    Args:
      eos:  instance of PengRobinsonEOS or any real-gas EOS with pressure, enthalpy
      rho1: upstream density [kg/m^3]
      T1:   upstream temperature [K]
      u1:   upstream velocity [m/s]

    Returns:
      (rho2, T2, p2, u2) for the post-shock state
    """
    p1 = eos.pressure(rho1, T1)
    h1 = eos.enthalpy(rho1, T1)

    # We'll guess a mild jump in density, T for starting fsolve
    rho2_guess = rho1*2.0
    T2_guess   = T1*1.1

    def froot(vars_):
        return shock_residual(vars_, eos, rho1, u1, h1, p1)

    sol = fsolve(froot, x0=[rho2_guess, T2_guess])
    rho2, T2 = sol
    p2 = eos.pressure(rho2, T2)
    u2 = (rho1*u1)/rho2

    return (rho2, T2, p2, u2)

# ------------------------------------------------------------------------
# (D) MAIN: Example usage
# ------------------------------------------------------------------------
def main():
    # 1) Initialize the EOS
    co2_eos = PengRobinsonEOS()

    # -------------------------------
    # USER INPUT: Upstream Conditions
    # (Example "transcritical" style conditions for CO2)
    # Values are arbitrary placeholders. You should input the EXACT data
    # from the Wang & Hickey (2020) example you want to replicate!
    # e.g. near-critical p1, T1, etc.
    T1   = 320.0   # K
    p1   = 4.0e6   # Pa
    # Let's guess a density from ideal gas or do a short solve
    # In a real scenario, you'd invert eos or do a separate routine.
    # For demonstration, approximate with p = rho R T => rho ~ p/(R T)
    # Then refine if needed:
    Rg   = co2_eos.Rg
    rho1_approx = p1/(Rg*T1)

    # We'll guess a velocity => Mach ~ 2.0? Let's do a guess
    # In a real scenario, you'd define Mach or velocity. We'll do 300 m/s here:
    u1 = 300.0

    # 2) Solve for post-shock state
    rho2, T2, p2, u2 = solve_normal_shock(co2_eos, rho1_approx, T1, u1)

    # 3) Print results
    print("\n--- Wang & Hickey Normal Shock Demo ---")
    print(f"Upstream guess:")
    print(f"  rho1 ~ {rho1_approx:.4f} kg/m^3, T1= {T1:.1f} K, p1 ~ {p1/1e6:.3f} MPa, u1= {u1:.1f} m/s")
    print(f"Downstream solution:")
    print(f"  rho2= {rho2:.4f} kg/m^3, T2= {T2:.1f} K")
    print(f"  p2= {p2/1e6:.3f} MPa, u2= {u2:.1f} m/s")

    # 4) Compare shock ratio or Mach ratio if needed
    # e.g. ratio
    p_ratio = p2/p1
    print(f"\nShock pressure ratio: p2/p1= {p_ratio:.3f}")
    # Done! If you want to compare to Wang & Hickey, adjust T1, p1, etc. to match their setup.


if __name__ == "__main__":
    main()
