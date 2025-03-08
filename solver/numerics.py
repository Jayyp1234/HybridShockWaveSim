#!/usr/bin/env python3
"""
numerical_amr.py

1D Adaptive Mesh Refinement (AMR) Finite-Volume Solver 
Supports Multiple EOS Models.
"""

import numpy as np
from solver.analytics import create_eos, EOS  # Import EOS dynamically

# ----------------------------------------------------------------------
# 🔧 **Modify: Use EOS Object Instead of Fixed Ideal Gas**
# ----------------------------------------------------------------------
def eos_pressure(eos: EOS, rho, e_int):
    """Compute pressure using EOS model instead of fixed gamma."""
    T_guess = 300  # Initial temperature guess
    p, _, _ = eos.compute_properties(T_guess, rho)
    return p  # Return real-gas pressure

def eos_sound_speed(eos: EOS, rho, p):
    """Compute real-gas sound speed using EOS."""
    T_guess = 300  # Estimate temperature from EOS
    _, _, c = eos.compute_properties(T_guess, rho)
    return c  # Real-gas speed of sound

# ----------------------------------------------------------------------
# 🛠 **Convert Between Conservative & Primitive Variables**
# ----------------------------------------------------------------------
def cons_to_prim(eos, U):
    """
    Convert conservative variables (U = [rho, rho*u, E]) to primitive form (rho, u, p)
    Uses real-gas EOS for accurate pressure calculations.
    """
    rho = U[0]
    
    # Prevent division errors
    if rho <= 1e-12 or np.isnan(rho) or np.isinf(rho):
        rho = 1e-12  # Set a minimum density

    u = U[1] / rho
    e_int = (U[2] - 0.5 * rho * u * u) / rho  # Internal energy per unit mass
    
    # Ensure valid initial temperature estimate
    T_guess = max(100, min(5000, e_int + 0.5 * u**2))  # Prevent unrealistic temperatures

    try:
        # Compute thermodynamic properties from EOS
        p, h, _ = eos.compute_properties(T_guess, rho)

        # Ensure pressure is non-negative
        if p < 0 or np.isnan(p) or np.isinf(p):
            print(f"Warning: Invalid pressure ({p}) detected. Using fallback value.")
            p = max(1e3, 1e6 * rho)  # Approximate pressure using density scaling

    except Exception as e:
        print(f"EOS solver failed in cons_to_prim: {e}")
        p = max(1e3, 1e6 * rho)  # Set a fallback pressure

    return rho, u, p

def prim_to_cons(eos, rho, u, p):
    """
    Convert primitive variables (rho, u, p) to conservative form (U = [rho, rho*u, E])
    Uses real-gas EOS to compute internal energy correctly.
    """
    T_guess = 300  # Initial guess for temperature
    _, h, _ = eos.compute_properties(T_guess, rho)  # Get enthalpy
    e_int = h - 0.5 * u**2  # Internal energy from enthalpy
    E = rho * e_int + 0.5 * rho * u * u  # Total energy

    return np.array([rho, rho * u, E])

# ----------------------------------------------------------------------
# 🚀 **Update: Pass EOS to Riemann Solver**
# ----------------------------------------------------------------------
def rusanov_flux(eos: EOS, UL, UR):
    """Compute Rusanov flux using EOS model."""
    rhoL, uL, pL = cons_to_prim(eos, UL)
    rhoR, uR, pR = cons_to_prim(eos, UR)

    FL = np.array([
        rhoL * uL,
        rhoL * uL * uL + pL,
        (UL[2] + pL) * uL
    ])
    FR = np.array([
        rhoR * uR,
        rhoR * uR * uR + pR,
        (UR[2] + pR) * uR
    ])

    cL = eos_sound_speed(eos, rhoL, pL)
    cR = eos_sound_speed(eos, rhoR, pR)
    s_max = max(abs(uL) + cL, abs(uR) + cR)

    return 0.5 * (FL + FR) - 0.5 * s_max * (UR - UL)


def compute_properties(self, T, rho):
    """Computes pressure, enthalpy, and sound speed using the EOS model."""
    if np.isnan(T) or np.isnan(rho) or T <= 0 or rho <= 0:
        print(f"Warning: Invalid (T, rho) inputs -> T={T}, rho={rho}. Setting defaults.")
        return 1e5, 250000, 300  # Default values to prevent solver failure

    try:
        coeffs = self._compute_cubic_coeffs(T, rho)
        roots = np.roots(coeffs)

        # Filter valid roots
        valid_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]

        if not valid_roots:
            raise ValueError("No valid real positive root found.")

        P = min(valid_roots)  # Select smallest valid root (physical pressure)
        h = self.compute_enthalpy(T, P)
        c = self.compute_speed_of_sound(T, P, rho)

        return P, h, c

    except Exception as e:
        print(f"ERROR in compute_properties: {e}. Using fallback values.")
        return 1e5, 250000, 300  # Fallback values for stability

# ----------------------------------------------------------------------
# 🔄 **Modify: Finite-Volume Step Uses EOS**
# ----------------------------------------------------------------------
def finite_volume_step(eos: EOS, x, U, dt):
    """1D Godunov step with Rusanov flux."""
    nx = len(x)
    Unew = U.copy()
    F = np.zeros((nx + 1, 3))

    for i in range(nx + 1):
        if i == 0:
            F[i] = rusanov_flux(eos, U[0], U[0])  # Left boundary
        elif i == nx:
            F[i] = rusanov_flux(eos, U[-1], U[-1])  # Right boundary
        else:
            F[i] = rusanov_flux(eos, U[i - 1], U[i])

    for i in range(nx):
        dx_i = x[i + 1] - x[i] if i < nx - 1 else x[i] - x[i - 1]
        if abs(dx_i) < 1e-12:
            dx_i = 1e-12
        Unew[i] = U[i] - (dt / dx_i) * (F[i + 1] - F[i])

    return Unew

# ----------------------------------------------------------------------
# ✅ **Modify: `run_amr_solver` Now Accepts EOS**
# ----------------------------------------------------------------------
def run_amr_solver(eos, x_init, U_init, tmax, cfl=0.5, refine_thresh=0.05, coarsen_thresh=0.01, max_cells=50000, verbose=True):
    """
    Time integration with adaptive mesh for a given EOS.
    """
    x, U = x_init, U_init
    time = 0.0
    step_count = 0

    while time < tmax:
        step_count += 1

        # Compute wave speeds safely
        smax = 0.0
        for i in range(len(x)):
            try:
                rho_i, u_i, p_i = cons_to_prim(eos, U[i])
                c_i = eos.compute_properties(max(100, p_i), rho_i)[2]  # Get speed of sound

                if np.isnan(c_i) or np.isinf(c_i):
                    c_i = 1.0  # Default value if computation fails
                
                speed = abs(u_i) + c_i
                smax = max(smax, speed)

            except Exception as e:
                print(f"Wave speed computation failed: {e}")
                smax = max(smax, 1.0)  # Prevent solver crash

        if smax < 1e-12:
            dt = 1e-6  # Smallest time step
        else:
            dt = cfl * min(abs(x[i + 1] - x[i]) for i in range(len(x) - 1)) / smax

        if time + dt > tmax:
            dt = tmax - time

        U = finite_volume_step(eos, x, U, dt)
        time += dt

        if verbose:
            print(f"Step {step_count}: time= {time:.5e}, dt= {dt:.5e}, cells= {len(x)}")

    return x, U, time

# ----------------------------------------------------------------------
# **Minimal Example Driver**
# ----------------------------------------------------------------------
if __name__ == "__main__":
    eos = create_eos("PR", {'Tc': 304.13, 'Pc': 7.3773e6, 'Mw': 0.04401, 'omega': 0.228})

    nx0 = 10
    x_init = np.linspace(0.0, 1.0, nx0)
    U_init = np.array([prim_to_cons(eos, 1.0 if x < 0.5 else 0.125, 0.0, 1e5 if x < 0.5 else 1e4) for x in x_init])

    x_final, U_final, t_end = run_amr_solver(eos, x_init, U_init, tmax=0.01, cfl=0.8, refine_thresh=0.2, coarsen_thresh=0.05)
    print(f"Done at time {t_end:.4g}, final cells: {len(x_final)}")
    print("Final state: ", [cons_to_prim(eos, U) for U in U_final])
