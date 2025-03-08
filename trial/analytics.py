#!/usr/bin/env python3
# analytics.py
"""
analytics.py

Provides multiple real-gas Equation of State (EOS) routines, each returning:
(density, enthalpy, a_sound)

Supported EOS:
1) Van der Waals (VDW)
2) Redlich–Kwong (RK)
3) Soave–Redlich–Kwong (SRK)
4) Peng–Robinson (PR)
5) Span–Wagner (SW) - placeholder
6) Virial (truncated) - placeholder

All use a numeric approach to invert p(rho,T) -> (T,P) for density.

Usage:
  from analytics import get_eos_model
  eos_func = get_eos_model("PR")
  rho, h, a = eos_func(T, P)

In a real scenario, you'd refine enthalpy and a_sound with proper
thermodynamic integrals, partial derivatives, and possibly handle
multiple roots in the cubic solver more carefully.
"""

import math
import numpy as np
from typing import Tuple

# ----------------------------------
# Constants & Example Fluid (CO2)
# ----------------------------------
R_UNIV    = 8.314462618  # J/(mol·K), universal constant
M_CO2     = 44.01e-3     # kg/mol for CO2
R_SPEC_CO2= R_UNIV / M_CO2

# Generic placeholders; adapt for actual fluid:
TC_CO2    = 304.1282     # K
PC_CO2    = 7.3773e6     # Pa
OMEGA_CO2 = 0.225        # accentric factor

# --------------------------------------------
#  (A) Van der Waals (VDW)
# --------------------------------------------
def invert_vdw_for_density(T: float, P: float, 
                           R_spec: float, Tc: float, Pc: float) -> float:
    """
    Numeric approach to solve Van der Waals:
      p = R_spec*T/(v - b) - a/(v^2)
    where a = 27(R_spec^2 Tc^2)/(64 Pc), b = R_spec*Tc/(8 Pc).

    Return density in kg/m^3.
    """
    a = 27.0 * (R_spec**2) * (Tc**2) / (64.0 * Pc)
    b = (R_spec * Tc) / (8.0 * Pc)

    def vdw_residual(vol: float) -> float:
        if vol <= 0.0:
            return 1e12
        term1 = (R_spec*T)/(vol - b)
        term2 = a/(vol**2)
        return term1 - term2 - P

    # Bisection or other bracket method
    v_low, v_high = 1e-6, 1.0
    for _ in range(100):
        v_mid = 0.5*(v_low + v_high)
        f_low = vdw_residual(v_low)
        f_mid = vdw_residual(v_mid)
        if f_low*f_mid <= 0:
            v_high = v_mid
        else:
            v_low = v_mid
        if abs(f_mid) < 1e-5:
            break

    v_sol = 0.5*(v_low + v_high)
    rho_sol = 1.0/v_sol
    return rho_sol

def eos_van_der_waals(T: float, P: float) -> Tuple[float,float,float]:
    """
    Return (rho, enthalpy, a_sound) using Van der Waals EOS.
    """
    rho = invert_vdw_for_density(T, P, R_SPEC_CO2, TC_CO2, PC_CO2)
    # approx enthalpy
    cp_approx = 1200.0
    h = cp_approx * T
    # approximate a_sound
    gamma_approx = 1.3
    a = math.sqrt(gamma_approx * R_SPEC_CO2 * T)
    return (rho, h, a)

# --------------------------------------------
#  (B) Redlich–Kwong (RK)
# --------------------------------------------
def invert_rk_for_density(T: float, P: float, 
                          R_spec: float, Tc: float, Pc: float) -> float:
    """
    p = R_spec*T/(v - b) - a/[v(v + b)*sqrt(T)]
    where a = 0.42748 (R_spec^2 Tc^{2.5}/Pc), b = 0.08664 R_spec Tc/Pc
    We'll do a bracket search for v in [1e-6, 1.0].
    """
    a = 0.42748 * (R_spec**2) * (Tc**2.5) / Pc
    b = 0.08664 * R_spec * Tc / Pc

    def rk_residual(vol: float) -> float:
        if vol <= 0:
            return 1e12
        term1 = (R_spec*T)/(vol - b)
        term2 = a / (vol*(vol + b)*math.sqrt(T))
        return term1 - term2 - P

    # Bisection
    v_low, v_high = 1e-6, 1.0
    for _ in range(100):
        v_mid = 0.5*(v_low + v_high)
        f_low = rk_residual(v_low)
        f_mid = rk_residual(v_mid)
        if f_low*f_mid <= 0:
            v_high = v_mid
        else:
            v_low = v_mid
        if abs(f_mid) < 1e-5:
            break

    v_sol = 0.5*(v_low + v_high)
    rho_sol = 1.0 / v_sol
    return rho_sol

def eos_redlich_kwong(T: float, P: float) -> Tuple[float,float,float]:
    rho = invert_rk_for_density(T, P, R_SPEC_CO2, TC_CO2, PC_CO2)
    cp_approx = 1200.0
    h = cp_approx * T
    gamma_approx = 1.3
    a = math.sqrt(gamma_approx * R_SPEC_CO2 * T)
    return (rho, h, a)

# --------------------------------------------
#  (C) Soave–Redlich–Kwong (SRK)
# --------------------------------------------
def invert_srk_for_density(T: float, P: float, 
                           R_spec: float, Tc: float, Pc: float, omega: float) -> float:
    """
    Soave modifies alpha(T) in RK with a (1 + k(1 - sqrt(T/Tc)))^2 approach.
    Then the EOS becomes:
      p = R_spec*T/(v - b) - a*alpha / [v(v + b)]
    where a = 0.42748 (R_spec^2 Tc^{2}/Pc),
          b = 0.08664 R_spec Tc / Pc,
          alpha = [1 + k(1 - sqrt(T/Tc))]^2,
          k = 0.48 + 1.574 omega - 0.176 omega^2
    """
    a0 = 0.42748 * (R_spec**2) * (Tc**2) / Pc
    b = 0.08664 * R_spec * Tc / Pc
    k = 0.48 + 1.574*omega - 0.176*(omega**2)
    alpha = (1 + k*(1 - math.sqrt(T/Tc)))**2
    a = a0 * alpha

    def srk_residual(vol: float) -> float:
        if vol <= 0:
            return 1e12
        term1 = (R_spec*T)/(vol - b)
        term2 = a / (vol*(vol + b))
        return term1 - term2 - P

    # bracket
    v_low, v_high = 1e-6, 1.0
    for _ in range(100):
        v_mid = 0.5*(v_low + v_high)
        f_low = srk_residual(v_low)
        f_mid = srk_residual(v_mid)
        if f_low*f_mid <= 0:
            v_high = v_mid
        else:
            v_low = v_mid
        if abs(f_mid) < 1e-5:
            break

    v_sol = 0.5*(v_low + v_high)
    rho_sol = 1.0 / v_sol
    return rho_sol

def eos_soave_rk(T: float, P: float) -> Tuple[float,float,float]:
    rho = invert_srk_for_density(T, P, R_SPEC_CO2, TC_CO2, PC_CO2, OMEGA_CO2)
    cp_approx = 1200.0
    h = cp_approx * T
    gamma_approx = 1.3
    a = math.sqrt(gamma_approx * R_SPEC_CO2 * T)
    return (rho, h, a)

# --------------------------------------------
#  (D) Peng–Robinson (PR)
# --------------------------------------------
def invert_peng_robinson_density(T: float, P: float, 
                                  R_spec: float, Tc: float, Pc: float, omega: float) -> float:
    """
    p = R_spec*T/(v - b) - a*alpha / (v^2 + 2*b*v - b^2)
    where a = 0.45724 (R_spec^2 Tc^2 / Pc),
          b = 0.07780 (R_spec Tc / Pc),
          alpha(T) = (1 + kappa*(1 - sqrt(T/Tc)))^2,
          kappa = 0.37464 + 1.54226*omega - 0.26992*omega^2
    """
    a0 = 0.45724 * (R_spec*Tc)**2 / Pc
    b = 0.07780 * (R_spec*Tc) / Pc
    kappa = 0.37464 + 1.54226*omega - 0.26992*(omega**2)
    alpha = (1 + kappa*(1 - math.sqrt(T/Tc)))**2
    a = a0 * alpha

    def pr_residual(vol: float) -> float:
        if vol <= 0:
            return 1e12
        term1 = (R_spec*T)/(vol - b)
        term2 = a / (vol**2 + 2*b*vol - b**2)
        return term1 - term2 - P

    # bracket
    v_low, v_high = 1e-6, 1.0
    for _ in range(100):
        v_mid = 0.5*(v_low + v_high)
        f_low = pr_residual(v_low)
        f_mid = pr_residual(v_mid)
        if f_low*f_mid <= 0:
            v_high = v_mid
        else:
            v_low = v_mid
        if abs(f_mid) < 1e-5:
            break

    v_sol = 0.5*(v_low + v_high)
    rho_sol = 1.0 / v_sol
    return rho_sol

def eos_peng_robinson(T: float, P: float) -> Tuple[float,float,float]:
    rho = invert_peng_robinson_density(T, P, R_SPEC_CO2, TC_CO2, PC_CO2, OMEGA_CO2)
    cp_approx = 1200.0
    h = cp_approx * T
    gamma_approx = 1.3
    a = math.sqrt(gamma_approx * R_SPEC_CO2 * T)
    return (rho, h, a)

# --------------------------------------------
#  (E) Span–Wagner (SW) - placeholder
# --------------------------------------------
def eos_span_wagner(T: float, P: float) -> Tuple[float,float,float]:
    """
    Real SW implementation is quite involved (dozens of coefficients).
    For demonstration, we'll do a naive call to invert PR or bisection, etc.
    Just to keep structure consistent.
    """
    rho_approx = invert_peng_robinson_density(T, P, R_SPEC_CO2, TC_CO2, PC_CO2, OMEGA_CO2)
    cp_approx = 1400.0
    h = cp_approx * T
    gamma_approx = 1.3
    a = math.sqrt(gamma_approx * R_SPEC_CO2 * T)
    return (rho_approx, h, a)

# --------------------------------------------
#  (F) Truncated Virial - placeholder
# --------------------------------------------
def eos_virial(T: float, P: float) -> Tuple[float,float,float]:
    """
    p = rho R_spec T (1 + B2(T)*rho), ignoring higher terms.

    We'll do a naive approach:
      p = rho R_spec T + rho^2 R_spec T B2(T)
    => solve for rho using bracket if possible.
    """
    # Placeholder B2(T) function. Real data would come from experiment or correlation.
    # Example: B2 ~ b0 - b1 / T
    b0, b1 = 1.0e-3, 1.0
    B2 = b0 - b1/T

    def virial_res(vol: float) -> float:
        if vol <= 0:
            return 1e12
        rr = 1.0 / vol
        # p_calc = rr*R_spec_co2*T * (1 + B2*rr)
        p_calc = rr*R_SPEC_CO2*T * (1 + B2*rr)
        return p_calc - P

    # bracket
    v_low, v_high = 1e-6, 1.0
    for _ in range(100):
        v_mid = 0.5*(v_low + v_high)
        f_low = virial_res(v_low)
        f_mid = virial_res(v_mid)
        if f_low*f_mid <= 0:
            v_high = v_mid
        else:
            v_low = v_mid
        if abs(f_mid) < 1e-5:
            break

    v_sol = 0.5*(v_low + v_high)
    rho_sol = 1.0 / v_sol

    cp_approx = 1200.0
    h = cp_approx*T
    gamma_approx = 1.3
    a = math.sqrt(gamma_approx * R_SPEC_CO2*T)
    return (rho_sol, h, a)


# --------------------------------------------
# (G) Factory function
# --------------------------------------------
def get_eos_model(eos_type: str = "PR"):
    """
    Usage:
       model_func = get_eos_model("VDW")
       rho, h, a = model_func( T, P )
    """
    eos_type = eos_type.upper()
    if eos_type == "VDW":
        return eos_van_der_waals
    elif eos_type == "RK":
        return eos_redlich_kwong
    elif eos_type == "SRK":
        return eos_soave_rk
    elif eos_type == "PR":
        return eos_peng_robinson
    elif eos_type == "SW":
        return eos_span_wagner
    elif eos_type == "VIRIAL":
        return eos_virial
    else:
        raise ValueError(f"Unknown EOS type: {eos_type}")


# --------------------------------------------
#  Demo if run as standalone
# --------------------------------------------
if __name__ == "__main__":
    # Quick tests
    T_demo, P_demo = 320.0, 5e6
    for eos_name in ["VDW", "RK", "SRK", "PR", "SW", "VIRIAL"]:
        func = get_eos_model(eos_name)
        rho, h, a = func(T_demo, P_demo)
        print(f"EOS = {eos_name}, T= {T_demo} K, P= {P_demo/1e6:.2f} MPa")
        print(f" -> rho= {rho:.4f} kg/m^3, h= {h:.1f} J/kg, a= {a:.2f} m/s\n")
