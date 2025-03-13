#!/usr/bin/env python3

"""
main.py

Runs the HybridShockWaveSim pipeline:
  1) Generates initial conditions from analytics.py
  2) Runs adaptive numerical solver (AMR-FVM) for multiple EOS models
  3) Validates results against experimental shock tube data
  4) Saves results for visualization in Bokeh
  5) Automatically launches Bokeh visualization

Usage:
  python main.py
"""

import pickle
import numpy as np
import pandas as pd
import sys
import threading
import matplotlib.pyplot as plt
import os

# Import numerical and analytical solvers
try:
    from solver.analytics import create_eos, ShockAnalyzer
    from solver.numerics import run_amr_solver, cons_to_prim, prim_to_cons
    from visualization.bokeh_plot import start_bokeh_server  # Auto-start visualization
except ImportError as e:
    print(f"Error: {e}")
    print("Ensure that solver/analytics.py, solver/numerics.py, and visualization/bokeh_plot.py are correctly structured.")
    sys.exit(1)

# Load experimental data
def load_experimental_data(filename="shock_experiment.csv"):
    """Loads experimental shock tube data for validation."""
    if not os.path.exists(filename):
        print(f"Error: Experimental data file '{filename}' not found!")
        sys.exit(1)
    
    data = pd.read_csv(filename)
    x_exp = data["x_exp"].values
    rho_exp = data["density_exp"].values
    p_exp = data["pressure_exp"].values  
    return x_exp, rho_exp, p_exp

# Compute L2 error between numerical and experimental results
def compute_validation_error(x_num, rho_num, p_num, x_exp, rho_exp, p_exp):
    """Computes L2 error norm between numerical and experimental profiles."""
    rho_exp_interp = np.interp(x_num, x_exp, rho_exp)  
    p_exp_interp = np.interp(x_num, x_exp, p_exp)
    
    error_rho = np.sqrt(np.mean((rho_exp_interp - rho_num) ** 2))
    error_p = np.sqrt(np.mean((p_exp_interp - p_num) ** 2))

    return error_rho, error_p

def main():
    """Runs the solver, validates it against experimental data, and starts Bokeh visualization."""

    # Define multiple EOS models to compare
    eos_models = {
        "PR": {'Tc': 304.13, 'Pc': 7.3773e6, 'Mw': 0.04401, 'omega': 0.228},
        "VDW": {'Tc': 304.13, 'Pc': 7.3773e6, 'Mw': 0.04401},
        "RK": {'Tc': 304.13, 'Pc': 7.3773e6, 'Mw': 0.04401},
        "SRK": {'Tc': 304.13, 'Pc': 7.3773e6, 'Mw': 0.04401, 'omega': 0.228}
    }

    # Load experimental data
    x_exp, rho_exp, p_exp = load_experimental_data()

    # Storage for all EOS results
    eos_results = {}

    for eos_name, params in eos_models.items():
        print(f"\nRunning simulation with {eos_name} EOS...")

        # Create EOS Model
        eos_model = create_eos(eos_name, params)
        analyzer = ShockAnalyzer(eos_model)

        # Set up initial conditions using Rankine-Hugoniot conditions
        pre_state = (1e6, 300, 100)  
        post_state = analyzer.rankine_hugoniot(pre_state)

        P2, T2, u2, rho2 = post_state

        print(f"{eos_name} EOS Shock Solution:")
        print(f"P2 = {P2/1e6:.2f} MPa, T2 = {T2:.1f} K, "
              f"u2 = {u2:.2f} m/s, rho2 = {rho2:.3f} kg/m^3")

        # Generate initial computational domain
        nx = 50  
        x_init = np.linspace(0, 1, nx)

        # Set up conservative variables **dynamically using EOS**
        U_init = np.zeros((nx, 3))
        for i in range(nx):
            if x_init[i] < 0.5:  # Left of shock
                rho, u, p = 1.0, 0.0, 1e5
            else:  # Right of shock (using computed EOS values)
                rho, u, p = rho2, u2, P2
            U_init[i] = prim_to_cons(eos_model, rho, u, p)
            

        # Run solver with EOS model **(Fixed Argument Order)**
        x_final, U_final, t_end = run_amr_solver(
            eos_model, x_init, U_init,  # **Ensure EOS model is first argument**
            tmax=0.0005, cfl=0.8,
            refine_thresh=0.2, coarsen_thresh=0.05,
            max_cells=50000, verbose=False
        )


        

        print(f"Solver completed at time {t_end:.5e}. Final cell count: {len(x_final)}")

        # Convert solver output
        rho_num = []
        p_num = []
        for i in range(len(x_final)):
            r, u, p = cons_to_prim(eos_model, U_final[i])  # Use EOS
            rho_num.append(r)
            p_num.append(p)

        # Compute validation error
        error_rho, error_p = compute_validation_error(x_final, rho_num, p_num, x_exp, rho_exp, p_exp)
        print(f"Validation ({eos_name}): L2 error -> Density = {error_rho:.5f}, Pressure = {error_p:.5f}")

        # Store results
        eos_results[eos_name] = {"x": x_final, "rho": rho_num, "p": p_num}

    # Store experimental data
    eos_results["experimental"] = {"x": x_exp, "rho": rho_exp, "p": p_exp}

    # Save results for visualization
    with open("shock_data.pkl", "wb") as f:
        pickle.dump(eos_results, f)

    print("Results saved to 'shock_data.pkl'. Bokeh visualization starting...")

    # Launch Bokeh visualization automatically
    def launch_bokeh():
        start_bokeh_server()

    # Start Bokeh in a separate thread to avoid blocking
    bokeh_thread = threading.Thread(target=launch_bokeh)
    bokeh_thread.start()

    # Plot DENSITY comparison
    plt.figure(figsize=(8, 5))
    for eos_name in eos_models:
        plt.plot(eos_results[eos_name]["x"], eos_results[eos_name]["rho"], linestyle="--", label=f"{eos_name} EOS")
        
    plt.scatter(x_exp, rho_exp, color="red", label="Experimental Data", marker="o")
    plt.xlabel("Position (x)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Shock Wave Density Validation (Multiple EOS)")
    plt.grid()
    plt.savefig("density_comparison.png")
    plt.show()

    # Plot PRESSURE comparison
    plt.figure(figsize=(8, 5))
    for eos_name in eos_models:
        plt.plot(eos_results[eos_name]["x"], eos_results[eos_name]["p"], linestyle="--", label=f"{eos_name} EOS")
    plt.scatter(x_exp, p_exp, color="blue", label="Experimental Data", marker="o")
    plt.xlabel("Position (x)")
    plt.ylabel("Pressure")
    plt.legend()
    plt.title("Shock Wave Pressure Validation (Multiple EOS)")
    plt.grid()
    plt.savefig("pressure_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
