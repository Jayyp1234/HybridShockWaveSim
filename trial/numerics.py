#!/usr/bin/env python3
# numerical.py
"""
A minimal 1D finite-volume solver demonstration.

Features:
  - initialize_domain(nx, x_start, x_end) : sets up the grid
  - finite_volume_step(state, dx, dt, eos_model) : performs one time-step
    using a simple Rusanov/HLL approximate Riemann solver

We store each cell's state in 'state' as:
  state[i, 0] = rho_i  (density)
  state[i, 1] = rho_i * u_i  (momentum)
  state[i, 2] = E_i    (total energy)

Usage (high level):
   x = initialize_domain(100, 0.0, 1.0)
   # define some initial condition in 'state'
   # pick an EOS model from your analytics or another module
   for n in range(num_steps):
       state = finite_volume_step(state, dx, dt, eos_model)

   # afterwards, convert final state to p, T, etc., for analysis
"""

import numpy as np

def initialize_domain(nx: int, x_start: float, x_end: float) -> np.ndarray:
    """
    Create a 1D spatial grid.
    Returns an array 'x' of cell-center coordinates, shape (nx,).
    """
    # For a finite-volume approach, we often consider cell centers (or edges).
    # We'll do cell centers below:
    return np.linspace(x_start, x_end, nx)

def compute_primitive(rho: float, mom: float, E: float, eos_model) -> tuple:
    """
    Convert conservative -> primitive (rho, u, p).
    We assume:
        E = rho * e + 0.5*rho*u^2   (total energy per cell)
    We need 'p' from the EOS, which might require a (rho, e) -> p step.

    Return (rho, u, p).
    """
    if rho <= 1e-12:
        rho = 1e-12

    u = mom / rho
    # internal energy per mass:
    e_int = (E - 0.5*rho*u*u) / rho
    if e_int < 0.0:
        # In robust code, handle negative e_int carefully (debug or fix).
        e_int = max(e_int, 1e-12)

    # With a real-gas EOS, we might invert e_int or (rho, e_int) to find T,
    # then do p = eos(rho, T). For demonstration, assume "p(rho, e_int)" is an EOS call.
    #
    # If your eos_model requires (T,P) -> (rho,h,a), you'd do an iterative approach
    # to find T s.t. e_int matches. We'll do a simpler approximate approach:
    # p ~ (gamma-1)*rho*e_int for an ideal-like placeholder.
    # Replace with a call to your real-gas function if needed.
    gamma_approx = 1.4
    p = (gamma_approx - 1.0)*rho*e_int

    # Alternatively, for a real-gas function: p = realgas_pressure(rho, e_int, eos_model)
    # e.g. p = eos_model.pressure(rho, e_int) if your model is written that way

    return (rho, u, p)

def compute_flux(rho: float, u: float, p: float, E: float) -> np.ndarray:
    """
    Return the flux vector F(U) = [rho*u, rho*u^2 + p, (E + p)*u].
    """
    return np.array([
        rho*u,
        rho*u*u + p,
        (E + p)*u
    ])

def riemann_flux(stateL: np.ndarray, stateR: np.ndarray, eos_model) -> np.ndarray:
    """
    A simple approximate Riemann solver for the interface flux.
    Using Rusanov (aka local Lax-Friedrichs):
      flux = 0.5*(F_L + F_R) - 0.5* s_max * (U_R - U_L)
    where s_max ~ max(|u_L|+c_L, |u_R|+c_R).

    stateL, stateR each has [rho, mom, E].
    eos_model is a pointer to EOS for computing p or speed of sound if needed.
    """
    rhoL, momL, EL = stateL
    rhoR, momR, ER = stateR

    # Convert to primitives
    rL, uL, pL = compute_primitive(rhoL, momL, EL, eos_model)
    rR, uR, pR = compute_primitive(rhoR, momR, ER, eos_model)

    # Speed of sound for left/right (approx or from real-gas):
    # We'll do c ~ sqrt(gamma * p/rho) for demonstration.
    # If using a real-gas EOS, call your function or do partial derivatives.
    gamma_approx = 1.4
    cL = np.sqrt(gamma_approx * pL / rL) if pL > 0.0 and rL > 0 else 0.0
    cR = np.sqrt(gamma_approx * pR / rR) if pR > 0.0 and rR > 0 else 0.0

    # Construct fluxes
    FL = compute_flux(rL, uL, pL, EL)
    FR = compute_flux(rR, uR, pR, ER)

    # wave speed estimate
    sL = abs(uL) + cL
    sR = abs(uR) + cR
    s_max = max(sL, sR)

    # Rusanov flux:
    flux = 0.5*(FL + FR) - 0.5*s_max*(stateR - stateL)
    return flux

def finite_volume_step(state: np.ndarray, dx: float, dt: float, eos_model) -> np.ndarray:
    """
    Perform one time-step of the finite-volume update (1D).
    state: shape (nx, 3) with columns [rho, rho*u, E]
    dx, dt: cell size, time step
    eos_model: function pointer to EOS (like from analytics.py)

    Return the updated state array (shape (nx,3)).
    """
    nx = state.shape[0]
    # We'll create a new array for the updated solution
    state_new = state.copy()

    # We need fluxes at each interface i+1/2
    # We'll do a loop from i=0 to i=nx-1 for cell edges
    # For boundary, assume a simple outflow or copy boundary states (transmissive).
    # More advanced codes do ghost cells, etc.
    fluxes = np.zeros((nx+1, 3))  # flux at each interface

    # 1) set boundary states (simple outflow)
    left_state  = state[0,:]
    right_state = state[nx-1,:]

    # 2) compute fluxes at each interface
    for i in range(nx+1):
        if i == 0:
            # left boundary
            UL = left_state
            UR = state[0,:]
        elif i == nx:
            # right boundary
            UL = state[nx-1,:]
            UR = right_state
        else:
            UL = state[i-1,:]
            UR = state[i,:]

        fluxes[i,:] = riemann_flux(UL, UR, eos_model)

    # 3) update each cell with flux difference
    # U^{n+1}_i = U^n_i - dt/dx * (F_{i+1/2} - F_{i-1/2})
    for i in range(nx):
        state_new[i,:] = state[i,:] - (dt/dx)*(fluxes[i+1,:] - fluxes[i,:])

    return state_new

# Optionally, we might define a function that runs multiple steps:
def run_simulation(state: np.ndarray, x: np.ndarray, cfl: float, total_time: float, eos_model):
    """
    Evolve the state up to 'total_time' using a maximum CFL of 'cfl'.
    """
    dx = x[1] - x[0]
    t = 0.0
    step_count = 0

    while t < total_time:
        # (1) compute s_max
        s_max = 0.0
        for i in range(state.shape[0]):
            rho_i, mom_i, E_i = state[i]
            # find p
            r_i, u_i, p_i = compute_primitive(rho_i, mom_i, E_i, eos_model)
            gamma_approx = 1.4
            c_i = np.sqrt(gamma_approx * p_i / r_i) if p_i>0 and r_i>0 else 0
            local_speed = abs(u_i) + c_i
            if local_speed > s_max:
                s_max = local_speed

        if s_max < 1e-12:
            # flow is almost at rest, can use a bigger dt
            dt = 1e-5
        else:
            dt = cfl * dx / s_max

        # (2) if dt would overshoot total_time
        if t + dt > total_time:
            dt = total_time - t

        # (3) do a single step
        state = finite_volume_step(state, dx, dt, eos_model)

        # (4) time increment
        t += dt
        step_count += 1

    print(f"Finished after {step_count} steps, final time {t}")
    return state

# Demo usage if run directly
if __name__ == "__main__":
    import numpy as np
    from analytics import get_eos_model

    nx = 50
    x = initialize_domain(nx, 0.0, 1.0)
    state = np.zeros((nx,3))
    # set initial conditions in 'state' ...
    eos_model = get_eos_model("PR")

    final = run_simulation(state, x, cfl=0.8, total_time=0.01, eos_model=eos_model)
    print("Done. final shape =", final.shape)
    # now convert final to p, T, etc., for analysis
    # e.g. for i in range(nx): p, T = compute_primitive(final[i,0], final[i,1], final[i,2], eos_model)