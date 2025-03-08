#!/usr/bin/env python3

"""
main.py

Runs the HybridShockWaveSim pipeline:
  1) Generates initial conditions from analytics.py
  2) Runs adaptive numerical solver (AMR-FVM)
  3) Validates results against experimental shock tube data
  4) Saves results for visualization in Bokeh

Usage:
  python main.py
"""

import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# Import numerical and analytical solvers
try:
    from solver.analytics import create_eos, ShockAnalyzer
    from solver.numerics import run_amr_solver, cons_to_prim
except ImportError as e:
    print(f"Error: {e}")
    print("Ensure that solver/analytics.py and solver/numerics.py are correctly structured.")
    sys.exit(1)

# Load experimental data
def load_experimental_data(filename="shock_experiment.csv"):
    """Loads experimental shock tube data for validation."""
    try:
        data = pd.read_csv(filename)
        x_exp = data["x_exp"].values
        rho_exp = data["density_exp"].values
        return x_exp, rho_exp
    except FileNotFoundError:
        print(f"Error: Experimental data file '{filename}' not found!")
        sys.exit(1)

# Compute L2 error between numerical and experimental results
def compute_validation_error(x_num, rho_num, x_exp, rho_exp):
    """Computes error norm between numerical and experimental density profiles."""
    rho_exp_interp = np.interp(x_num, x_exp, rho_exp)  # Interpolates experimental data
    error = np.sqrt(np.sum((rho_exp_interp - rho_num) ** 2) / len(rho_num))  # L2 norm
    return error

def main():
    """Runs the solver and validates it against experimental data."""

    # 1) Select an EOS model
    eos_type = "PR"  # Change to "VDW", "RK", "SRK", "SW", "VIRIAL", or "IDEAL" as needed
    params = {'Tc': 304.13, 'Pc': 7.3773e6, 'Mw': 0.04401, 'omega': 0.228}
    eos_model = create_eos(eos_type, params)
    analyzer = ShockAnalyzer(eos_model)

    # 2) Set up initial conditions using Rankine-Hugoniot conditions
    pre_state = (1e6, 300, 100)  # Example: P1=1MPa, T1=300K, u1=100m/s
    post_state = analyzer.rankine_hugoniot(pre_state)
    
    print(f"\n{eos_type} EOS Shock Solution:")
    print(f"P2 = {post_state[0]/1e6:.2f} MPa, T2 = {post_state[1]:.1f} K, "
          f"u2 = {post_state[2]:.2f} m/s, rho2 = {post_state[3]:.3f} kg/m^3")

    # 3) Generate initial computational domain
    nx = 50  # Initial number of cells
    x_init = np.linspace(0, 1, nx)

    # 4) Set up conservative variables (U = [rho, rho*u, E])
    U_init = np.zeros((nx, 3))
    for i in range(nx):
        if x_init[i] < 0.5:  # Left of shock
            rho, u, p = 1.0, 0.0, 1e5
        else:  # Right of shock
            rho, u, p = 0.125, 0.0, 1e4
        U_init[i] = np.array([rho, rho * u, rho * (p / ((1.4 - 1) * rho) + 0.5 * u * u)])

    # 5) Run numerical solver with AMR
    x_final, U_final, t_end = run_amr_solver(
        x_init, U_init,
        tmax=0.0005, cfl=0.8,
        refine_thresh=0.2, coarsen_thresh=0.05,
        gamma=1.4, max_cells=50000, verbose=True
    )

    print(f"\nSolver completed at time {t_end:.5e}. Final cell count: {len(x_final)}")

    # 6) Convert solver output to density & pressure
    rho_num = []
    pressure_num = []
    for i in range(len(x_final)):
        r, u, p = cons_to_prim(U_final[i], gamma=1.4)
        rho_num.append(r)
        pressure_num.append(p)

    # 7) Load experimental data
    x_exp, rho_exp = load_experimental_data()


    # 8) Validate numerical solution against experiment
    error = compute_validation_error(x_final, rho_num, x_exp, rho_exp)
    print(f"Validation: L2 error between numerical and experimental density = {error:.5f}")

    # 9) Save results for visualization in Bokeh
    with open("shock_data.pkl", "wb") as f:
        pickle.dump((x_final, rho_num, pressure_num, x_exp, rho_exp), f)
    
    print("Saved results to 'shock_data.pkl'. Run 'bokeh serve --show bokeh_plot.py' to visualize.")

    # 10) Plot numerical vs experimental comparison
    plt.figure(figsize=(8,5))
    plt.plot(x_final, rho_num, label="Numerical Solution", linestyle="--")
    plt.scatter(x_exp, rho_exp, color="red", label="Experimental Data", marker="o")
    plt.xlabel("Position (x)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Shock Wave Density Validation")
    plt.grid()
    plt.savefig("validation_plot.png")  # Save as image
    plt.show()

if __name__ == "__main__":
    main()
