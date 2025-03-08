#!/usr/bin/env python3
"""
numerical_amr.py

Demonstrates a 1D finite-volume solver with adaptive mesh refinement (AMR).

Key Components:
  1) Finite-volume step (Godunov-type / Rusanov flux).
  2) Adaptive mesh refinement:
     - After each time-step, we check each cell's gradient (e.g., density gradient).
     - If gradient > refine_threshold, we split that cell into two.
     - If gradient < coarsen_threshold, we merge adjacent cells.
  3) A simple example to show how everything ties together.

Caveats:
  - The code is minimal and for demonstration. Real AMR solvers handle
    multi-level grids, ghost cells, flux redistribution, etc.
  - The Rusanov flux is robust but not very sharp. For better shock resolution,
    you could use HLLC, Roe, or add slope limiters / FCT.
  - Data mapping between old and refined/coarsened grids is simplistic.

Dependencies:
  - numpy

Usage Example (command line):
  python numerical_amr.py

You can embed this in your larger project or import the classes below.
"""

import numpy as np

# ----------------------------------------------------------------------
# Example: A trivial equation of state for demonstration
# Real usage: you might import from your analytics.py or real-gas classes
# ----------------------------------------------------------------------
def eos_pressure(rho, e_int, gamma=1.4):
    """
    p = (gamma-1)*rho*e
    Ideal gas approach, just for demonstration
    """
    return (gamma - 1.0)*rho*e_int

def eos_sound_speed(rho, p, gamma=1.4):
    """
    c = sqrt(gamma * p / rho).
    """
    if rho <= 1e-12 or p <= 0:
        return 0.0
    return np.sqrt(gamma * p / rho)

# ----------------------------------------------------------------------
# Helper: Convert between Conservative (U) and Primitive (rho,u,p)
# ----------------------------------------------------------------------
def cons_to_prim(U, gamma=1.4):
    """
    U = [rho, mom, E]
    returns (rho, u, p)
    where
    E = rho*e + 0.5*rho*u^2
    e = (E - 0.5*rho*u^2)/rho
    p = (gamma-1)*rho*e
    """
    rho = U[0]
    if rho < 1e-12:
        rho = 1e-12  # avoid zero-div issues
    u = U[1]/rho
    e_int = (U[2] - 0.5*rho*u*u)/rho
    p = eos_pressure(rho, e_int, gamma=gamma)
    return (rho, u, p)

def prim_to_cons(rho, u, p, gamma=1.4):
    """
    from (rho, u, p) -> U = [rho, rho*u, E]
    E = rho*e + 0.5*rho*u^2
    e = p / ((gamma-1)*rho)
    """
    e_int = p / ((gamma - 1.0)*rho)
    E = rho*e_int + 0.5*rho*u*u
    return np.array([rho, rho*u, E])

# ----------------------------------------------------------------------
# Riemann Solver: Rusanov (Local Lax-Friedrichs)
# ----------------------------------------------------------------------
def rusanov_flux(UL, UR, gamma=1.4):
    """
    UL, UR: [rho, mom, E]
    Returns flux vector F(3).
    """
    # convert to primitives
    rhoL, uL, pL = cons_to_prim(UL, gamma=gamma)
    rhoR, uR, pR = cons_to_prim(UR, gamma=gamma)

    # compute fluxes on each side
    FL = np.array([
        rhoL*uL,
        rhoL*uL*uL + pL,
        (UL[2] + pL)*uL
    ])
    FR = np.array([
        rhoR*uR,
        rhoR*uR*uR + pR,
        (UR[2] + pR)*uR
    ])

    # wave speeds
    cL = eos_sound_speed(rhoL, pL, gamma=gamma)
    cR = eos_sound_speed(rhoR, pR, gamma=gamma)
    s_max = max(abs(uL) + cL, abs(uR) + cR)

    # Rusanov
    flux = 0.5*(FL + FR) - 0.5*s_max*(UR - UL)
    return flux

# ----------------------------------------------------------------------
# Finite-Volume Step in 1D
# ----------------------------------------------------------------------
def finite_volume_step(x, U, dt, gamma=1.4):
    """
    1D Godunov step with Rusanov flux.
    x, U: 1D arrays of cell centers and states
    shape of U: (nx, 3)
    dt: time step
    returns updated U
    """
    nx = len(x)
    Unew = U.copy()
    # fluxes at interfaces
    F = np.zeros((nx+1, 3))

    # simple boundary states (outflow)
    UL_bound = U[0]
    UR_bound = U[-1]

    # 1) compute flux at each interface
    for i in range(nx+1):
        if i == 0:
            # left boundary
            FL_ = UL_bound
            FR_ = U[0]
        elif i == nx:
            FL_ = U[-1]
            FR_ = UR_bound
        else:
            FL_ = U[i-1]
            FR_ = U[i]
        F[i] = rusanov_flux(FL_, FR_, gamma=gamma)

    # 2) update each cell
    for i in range(nx):
        if nx>1:
            if i < nx - 1:
                dx_i = x[i+1] - x[i]
            else:
                dx_i = x[i] - x[i-1] if i>0 else 0.01
        else:
            # if only 1 cell, define an artificial dx
            dx_i = x[0] if len(x)>0 else 0.01

        if abs(dx_i) < 1e-12:
            dx_i = 1e-12

        Unew[i] = U[i] - (dt/dx_i)*(F[i+1] - F[i])

    return Unew

# ----------------------------------------------------------------------
# Adaptive Mesh Refinement
# ----------------------------------------------------------------------
def refine_coarsen(x, U, refine_thresh=0.05, coarsen_thresh=0.01, gamma=1.4):
    """
    1D adaptation: if density gradient > refine_thresh, we split cell.
    if < coarsen_thresh, we try merging.

    x: array of cell centers, shape (nx,)
    U: array of shape (nx,3)
    returns (x_new, U_new)
    """
    nx = len(x)
    new_x = []
    new_U = []

    i = 0
    while i < nx:
        # keep boundary cells
        if i == 0 or i == nx - 1:
            new_x.append(x[i])
            new_U.append(U[i])
            i += 1
            continue

        # compute density gradient w.r.t previous cell
        rho_curr, _, _ = cons_to_prim(U[i], gamma=gamma)
        rho_prev, _, _ = cons_to_prim(U[i-1], gamma=gamma)
        dx_i = x[i] - x[i-1]
        if abs(dx_i) < 1e-12:
            dx_i = 1e-12
        grad_rho = abs(rho_curr - rho_prev)/dx_i

        if grad_rho > refine_thresh:
            # refine: split cell i into two
            x_left = x[i] - 0.25*dx_i
            x_right= x[i] + 0.25*dx_i
            U_half = 0.5*U[i]  # naive

            new_x.append(x_left)
            new_U.append(U_half.copy())

            new_x.append(x_right)
            new_U.append(U_half.copy())

            i += 1
        elif grad_rho < coarsen_thresh and i < nx-1:
            # coarsen: merge cell i and i+1
            mergedU = U[i] + U[i+1]
            mergedX = 0.5*(x[i] + x[i+1])
            new_x.append(mergedX)
            new_U.append(mergedU)
            i += 2
        else:
            # keep cell
            new_x.append(x[i])
            new_U.append(U[i])
            i += 1

    x_new = np.array(new_x)
    U_new = np.array(new_U)

    # ensure sorted
    idx = np.argsort(x_new)
    x_new = x_new[idx]
    U_new = U_new[idx]
    return x_new, U_new

# ----------------------------------------------------------------------
# Example driver: run_amr_solver
# ----------------------------------------------------------------------
def run_amr_solver(x_init, U_init, tmax,
                   cfl=0.5, refine_thresh=0.05, coarsen_thresh=0.01,
                   gamma=1.4, max_cells=50000, verbose=True):
    """
    Simple driver for time integration with adaptive mesh after each step.
    x_init, U_init: initial mesh & state
    tmax: final time
    cfl: Courant number
    refine_thresh, coarsen_thresh: AMR triggers
    gamma: ratio of specific heats
    max_cells: maximum number of cells allowed
    verbose: if True, print iteration details
    """
    x, U = x_init, U_init
    time = 0.0
    step_count = 0

    while time < tmax:
        step_count += 1

        # 1) compute dt from wave speeds
        smax = 0.0
        for i in range(len(x)):
            rho_i, u_i, p_i = cons_to_prim(U[i], gamma=gamma)
            c_i = eos_sound_speed(rho_i, p_i, gamma=gamma)
            speed = abs(u_i) + c_i
            smax = max(smax, speed)

        if smax < 1e-12:
            dt = 1e-6
        else:
            # find minimal dx
            avg_dx = 1e9
            for j in range(len(x)-1):
                dxj = abs(x[j+1] - x[j])
                if dxj < avg_dx:
                    avg_dx = dxj
            dt = cfl * avg_dx / smax

        # adjust dt if overshooting final time
        if time + dt > tmax:
            dt = tmax - time

        # 2) finite-volume step
        U = finite_volume_step(x, U, dt, gamma=gamma)
        time += dt

        # 3) refine / coarsen
        x, U = refine_coarsen(x, U, refine_thresh=refine_thresh,
                              coarsen_thresh=coarsen_thresh, gamma=gamma)

        # 4) Check max cell limit
        if len(x) > max_cells:
            # We skip further refinement. One approach: revert to the unrefined mesh
            # or simply do nothing. The simplest: don't refine if we exceed the limit.
            # We'll forcibly coarsen or skip refine. Here's a naive approach:
            x = x[:max_cells]   # or keep partial? There's no perfect solution
            U = U[:max_cells]
            # Or just break if you prefer to stop the simulation:
            break
            # Alternatively, to be more sophisticated: 
            # You might revert to a version of x, U from before refinement 
            # or set refine_thresh to a huge value so it stops refining further.

        # If verbose, print iteration info
        if verbose:
            # compute min/max density
            rho_vals = []
            for i in range(len(x)):
                r_i, u_i, p_i = cons_to_prim(U[i], gamma=gamma)
                rho_vals.append(r_i)
            min_rho, max_rho = min(rho_vals), max(rho_vals)

            print(f"Step {step_count}: time= {time:.5e}, dt= {dt:.5e}, "
                  f"smax= {smax:.5e}, cells= {len(x)}, "
                  f"minRho= {min_rho:.5e}, maxRho= {max_rho:.5e}")

    return x, U, time

# ----------------------------------------------------------------------
# Minimal Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # initial domain
    nx0 = 10
    x_start, x_end = 0.0, 1.0
    x_init = np.linspace(x_start, x_end, nx0)

    # initial condition: e.g. a shock tube
    # U[i] = [rho, rho*u, E]
    # left: rho=1, p=1e5, right: rho=0.125, p=1e4
    U_init = np.zeros((nx0,3))
    for i in range(nx0):
        if x_init[i]<0.5*(x_start+x_end):
            rho, u, p = 1.0, 0.0, 1e5
        else:
            rho, u, p = 0.125, 0.0, 1e4
        U_init[i] = prim_to_cons(rho, u, p, gamma=1.4)

    # run
    tmax = 0.01
    x_final, U_final, t_end = run_amr_solver(
        x_init, U_init, tmax,
        cfl=0.8,
        refine_thresh=0.2,
        coarsen_thresh=0.05,
        gamma=1.4
    )
    print(f"Done at time {t_end:.4g}")
    print("Final number of cells:", len(x_final))

    # show final density
    rho_fin = []
    for i in range(len(x_final)):
        r, u, p = cons_to_prim(U_final[i], gamma=1.4)
        rho_fin.append(r)
    # sort by x
    idx_sort = np.argsort(x_final)
    x_plot = x_final[idx_sort]
    rho_plot = np.array(rho_fin)[idx_sort]

    for xx, rr in zip(x_plot, rho_plot):
        print(f"x= {xx:.4f}, rho= {rr:.4f}")
