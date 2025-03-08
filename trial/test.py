#!/usr/bin/env python3
"""
Analytical solutions to shocks and rarefactions for non-ideal state equations
Python version of normal_shock_v3.m

Author: [Your Name]
Last Update: 2020-02-29 (MATLAB) / [Today’s Date] (Python conversion)

Description:
    - Computes normal shock relations for ideal and non-ideal gases (using RK, SRK, PR EoS)
    - Valid only for pure gases with no phase change.
    - Includes sensitivity analysis and plotting.
    
NOTE:
    This code is a direct translation of the MATLAB code. Some differences in indexing and
    plotting appear due to language differences.
    
    IMPORTANT: For the non-ideal cases (eos 1, 2, 3), the symbolic integration of several
    expressions (exp_intdpdT, exp_intvdpdv, exp_intd2pdT2) was replaced with manual formulas
    as given in the MATLAB code comments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve

# -------------------------
# INITIALIZATION & CONSTANTS
# -------------------------
print("Normal shock calculation initiated...")

# Universal constant [J kmol^-1 K^-1]
R = 8314.4621

# -------------------------
# INPUTS
# -------------------------
print("Reading spreadsheets with runs and gas properties data...")

# Input file names (ensure these files are in your working directory)
filename_runs = 'normal_shock_runs_constp.xlsx'
filename_properties = 'gas_properties.xlsx'

# fsolve parameters
iterations_fsolve_max = 10000
prefactor = 1  # Do not change unless you have a reason to use a different initial guess

# Display flags for optional plotting
show_fsolve = True
show_graph_sensitivity_cpcv = False
show_graph_sensitivity_c = False
show_graph_sensitivity_h_IG = False
show_graph_sensitivity_deltah = False
show_graph_sensitivity_h = False
show_graph_sensitivity_goveqn = False
show_graphs_final = True

# Graph colors and settings
color_ideal = [0, 0, 1]
color_RK = [0, 0.5, 0]
color_SRK = [0.6350, 0.0780, 0.1840]
color_PR = [0.75, 0, 0.75]
size_font = 11
size_font_sensitivity = 11
xlim_low = 1
xlim_high = 4
# For plotting markers – note that Python is 0-indexed
index_SUBC = 0  # (MATLAB index 1 becomes 0 in Python)
index_TRAC = 4  # (MATLAB index 5 becomes 4)
index_SUPC = 9  # (MATLAB index 10 becomes 9)

# Read Excel input files using pandas
runs_df = pd.read_excel(filename_runs)
properties_df = pd.read_excel(filename_properties)

print("\n-------------------------------------------------------------")
print("-------------------------------------------------------------\n")

# Define symbolic variables
p_sym, v_sym, T_sym = sp.symbols('p v T')

# Dictionary to store computed results (optional)
results = {}

# -------------------------
# LOOP OVER EQUATIONS OF STATE (EoS)
# -------------------------
# We loop over eos values 1:3 corresponding to RK, SRK, PR.
for eos in [1, 2, 3]:
    for run_idx, run in runs_df.iterrows():
        print(f"Now processing Run #{run_idx+1} of {len(runs_df)}...")
        
        # -------------------------
        # PROCESSING INPUTS
        # -------------------------
        gas_name = run['gas_name']
        # Retrieve gas properties (first row matching gas_name)
        prop = properties_df[properties_df['gas_name'] == gas_name].iloc[0]
        gamma_val = prop['gamma']
        M_val = prop['M']
        T_c_val = prop['T_c']
        p_c_val = prop['p_c']
        omega_val = prop['omega']
        h_ref_val = prop['h_ref']
        
        # Retrieve fit parameters for thermally perfect c_IG_p.
        # Remove commas from tokens (e.g., "3.259," -> "3.259")
        A_vals = np.array([float(x.replace(',', '')) for x in str(prop['c_IG_p_fit_params']).split()])
        
        # Retrieve pre-shock conditions from current run
        T1 = run['T_1']
        rho1 = run['rho_1']
        u1 = run['u_1']
        # Sensitivity analysis parameters
        upper_lim_factor = run['upper_lim_factor']
        lower_lim_factor = run['lower_lim_factor']
        num_intervals_per_var = int(run['num_intervals_per_var'])
        
        # Calculate the gas-specific constant [J kg^-1 K^-1]
        R_specific = R / M_val
        
        # -------------------------
        # SYMBOLIC DEFINITIONS & EoS SETUP
        # -------------------------
        if eos == 0:
            # Ideal Gas (not used in this loop)
            EoS_tag = "Ideal and thermally perfect gas"
            a_val = 0
            b_val = 0
            Theta_expr = 0
            delta_val = 0
            epsilon_val = 0
            p_expr = R*T_sym/(v_sym)
        elif eos == 1:
            EoS_tag = "RK with departure functions relative to thermally perfect gas"
            a_val = 0.4278 * R**2 * T_c_val**2.5 / p_c_val
            b_val = 0.0867 * R * T_c_val / p_c_val
            Theta_expr = a_val * (1/T_sym)**sp.Rational(1, 2)
            delta_val = b_val
            epsilon_val = 0
            p_expr = R*T_sym/(v_sym - b_val) - Theta_expr/(v_sym**2 + delta_val*v_sym)
        elif eos == 2:
            EoS_tag = "SRK with departure functions relative to thermally perfect gas"
            a_val = 0.42747 * R**2 * T_c_val**2 / p_c_val
            b_val = 0.08664 * R * T_c_val / p_c_val
            Theta_expr = a_val * (1 + (0.48 + 1.574*omega_val - 0.176*omega_val**2) * (1 - (T_sym/T_c_val)**sp.Rational(1, 2)))**2
            delta_val = b_val
            epsilon_val = 0
            p_expr = R*T_sym/(v_sym - b_val) - Theta_expr/(v_sym**2 + delta_val*v_sym)
        elif eos == 3:
            EoS_tag = "PR with departure functions relative to thermally perfect gas"
            a_val = 0.45724 * R**2 * T_c_val**2 / p_c_val
            b_val = 0.07780 * R * T_c_val / p_c_val
            Theta_expr = a_val * (1 + (0.37464 + 1.54226*omega_val - 0.2699*omega_val**2) * (1 - (T_sym/T_c_val)**sp.Rational(1, 2)))**2
            delta_val = 2 * b_val
            epsilon_val = -b_val**2
            p_expr = R*T_sym/(v_sym - b_val) - Theta_expr/(v_sym**2 + delta_val*v_sym + epsilon_val)
        
        print(f"\tEoS selection: {EoS_tag}...")
        
        # Compute symbolic derivatives
        dThetadT = sp.diff(Theta_expr, T_sym)
        d2ThetadT2 = sp.diff(Theta_expr, T_sym, 2)
        exp_dpdT = sp.diff(p_expr, T_sym)
        exp_d2pdT2 = sp.diff(p_expr, T_sym, 2)
        exp_dpdv = sp.diff(p_expr, v_sym)
        
        # -------------------------
        # MANUAL INTEGRATION EXPRESSIONS
        # -------------------------
        # For non-ideal EoS we replace the symbolic integration with the manual forms.
        # (Based on the MATLAB code's comments.)
        sqrt_term = sp.sqrt(delta_val**2 - 4*epsilon_val) if (delta_val**2 - 4*epsilon_val) != 0 else delta_val
        exp_intdpdT_expr = R*sp.log(v_sym - b_val) + dThetadT * sp.log((sqrt_term + 2*v_sym + delta_val) / (sqrt_term - 2*v_sym - delta_val)) / sqrt_term
        exp_intvdpdv_expr = (-Theta_expr*v_sym/(v_sym**2 + delta_val*v_sym + epsilon_val)
                             - Theta_expr * sp.log((sqrt_term + 2*v_sym + delta_val) / (sqrt_term - 2*v_sym - delta_val)) / sqrt_term
                             - R*T_sym*sp.log(v_sym - b_val) + R*T_sym*b_val/(v_sym - b_val))
        exp_intd2pdT2_expr = d2ThetadT2 * sp.log((sqrt_term + 2*v_sym + delta_val) / (sqrt_term - 2*v_sym - delta_val)) / sqrt_term
        
        # Specific heat capacities and enthalpies (ideal gas, thermally perfect)
        c_p_IG_TP_expr = R * sum([A_vals[i] * T_sym**i for i in range(len(A_vals))])
        c_v_IG_TP_expr = c_p_IG_TP_expr - R
        c_v_expr = c_v_IG_TP_expr + T_sym * exp_intd2pdT2_expr
        c_p_expr = c_v_expr - T_sym * (exp_dpdT**2) / exp_dpdv
        
        # Enthalpy departure function using the manual integration expression
        delta_h_IG_TP_expr = (Theta_expr - T_sym*dThetadT) * sp.log((sqrt_term+2*v_sym+delta_val)/(sqrt_term-2*v_sym-delta_val)) / sqrt_term + R*T_sym - p_expr*v_sym
        if eos == 0:  # Ideal gas case (not used here)
            delta_h_IG_TP_expr = 0
        h_IG_TP_expr = sp.integrate(c_p_IG_TP_expr, T_sym) - h_ref_val
        h_expr = h_IG_TP_expr - delta_h_IG_TP_expr
        
        # Miller & Bellan departure for Peng-Robinson (if desired)
        K_1_expr = (1 / (2 * sp.sqrt(2) * b_val)) * sp.log((v_sym + (1 - sp.sqrt(2)) * b_val) / (v_sym + (1 + sp.sqrt(2)) * b_val))
        delta_h_miller_expr = - p_expr*v_sym + R*T_sym - K_1_expr*(Theta_expr - T_sym*dThetadT)
        h_miller_expr = h_IG_TP_expr - delta_h_miller_expr
        
        # Speed of sound expression
        c_expr = sp.sqrt(-v_sym**2 * c_p_expr * exp_dpdv / (M_val * sp.simplify(c_v_expr)))
        
        # -------------------------
        # PRE-SHOCK COMPUTATIONS
        # -------------------------
        print("\tComputing pre-shock conditions...")
        # Pre-shock: set T = T1 and compute molar volume v1
        v1 = M_val / rho1
        
        subs_pre = {T_sym: T1, v_sym: v1}
        p_1_IG = rho1 * R_specific * T1
        c_p_1_IG_TP = float(c_p_IG_TP_expr.subs(subs_pre))
        c_v_1_IG_TP = float(c_v_IG_TP_expr.subs(subs_pre))
        c_1_IG_CP = np.sqrt(gamma_val * R * T1 / M_val)
        Ma_1_IG_CP = u1 / c_1_IG_CP
        h_1_IG_TP = float(h_IG_TP_expr.subs(subs_pre))
        
        # Real-gas pre-shock evaluations
        exp_dpdT_1 = float(exp_dpdT.subs(subs_pre))
        exp_dpdv_1 = float(exp_dpdv.subs(subs_pre))
        exp_d2pdT2_1 = float(exp_d2pdT2.subs(subs_pre))
        exp_intdpdT_1 = float(sp.re(exp_intdpdT_expr.subs(subs_pre)))
        exp_intvdpdv_1 = float(sp.re(exp_intvdpdv_expr.subs(subs_pre)))
        exp_intd2pdT2_1 = float(sp.re(exp_intd2pdT2_expr.subs(subs_pre)))
        p_1 = float(p_expr.subs(subs_pre))
        c_p_1 = float(sp.re(c_p_expr.subs(subs_pre)))
        c_v_1 = float(sp.re(c_v_expr.subs(subs_pre)))
        c_1 = float(sp.re(c_expr.subs(subs_pre)))
        Ma_1 = u1 / c_1
        delta_h_1_IG_TP = float(sp.re(delta_h_IG_TP_expr.subs(subs_pre)))
        delta_h_1_miller = float(delta_h_miller_expr.subs(subs_pre))
        h_1 = float(sp.re(h_expr.subs(subs_pre)))
        
        # -------------------------
        # POST-SHOCK (IDEAL GAS) COMPUTATIONS
        # -------------------------
        print("\tComputing post-shock conditions (ideal gas)...")
        p2p1_IG_CP = 1 + 2 * gamma_val * (Ma_1_IG_CP**2 - 1) / (gamma_val + 1)
        v2v1_IG_CP = (2 + (gamma_val - 1) * Ma_1_IG_CP**2) / ((gamma_val + 1) * Ma_1_IG_CP**2)
        T2T1_IG_CP = p2p1_IG_CP * v2v1_IG_CP
        u2u1_IG_CP = v2v1_IG_CP  # From continuity
        p_2_IG_CP = p2p1_IG_CP * p_1_IG
        v_2_IG_CP = v2v1_IG_CP * v1
        T_2_IG_CP = T2T1_IG_CP * T1
        u_2_IG_CP = u2u1_IG_CP * u1
        c_2_IG_CP = np.sqrt(gamma_val * R * T_2_IG_CP / M_val)
        Ma_2_IG_CP = u_2_IG_CP / c_2_IG_CP
        Ma2Ma1_IG_CP = Ma_2_IG_CP / Ma_1_IG_CP
        
        # -------------------------
        # POST-SHOCK (REAL GAS) COMPUTATIONS VIA FSOLVE
        # -------------------------
        print("\tComputing post-shock conditions (real gas)...")
        # Define the nonlinear normal shock equations symbolically
        eqn_ns_1 = p_expr - p_1 - (u1**2 / ((v1/M_val)**2)) * ((v1/M_val) - (v_sym/M_val))
        eqn_ns_2 = (h_expr - h_1) / M_val - (u1**2 / 2) * (1 - v_sym**2 / v1**2)
        
        # Create lambdified functions (with T and v as variables)
        f1 = sp.lambdify((T_sym, v_sym), eqn_ns_1, 'numpy')
        f2 = sp.lambdify((T_sym, v_sym), eqn_ns_2, 'numpy')
        
        def f_system(x):
            T_val, v_val = x
            return [f1(T_val, v_val), f2(T_val, v_val)]
        
        # Use ideal gas post-shock conditions as initial guess
        x0 = [prefactor * T_2_IG_CP, prefactor * v_2_IG_CP]
        if show_fsolve:
            sol = fsolve(f_system, x0, xtol=1e-12, maxfev=iterations_fsolve_max, full_output=True)
            S = sol[0]
            print("fsolve iterations:", sol[1]['nfev'])
        else:
            S = fsolve(f_system, x0, xtol=1e-12, maxfev=iterations_fsolve_max)
        T_2 = S[0]
        v_2 = S[1]
        
        subs_post = {T_sym: T_2, v_sym: v_2}
        p_2 = float(p_expr.subs(subs_post))
        c_2 = float(sp.re(c_expr.subs(subs_post)))
        p2p1 = p_2 / p_1
        v2v1 = v_2 / v1
        T2T1 = T_2 / T1
        u2u1 = v2v1  # by continuity
        u_2 = u1 * u2u1
        Ma_2 = u_2 / c_2
        Ma2Ma1 = Ma_2 / Ma_1
        c_p_2_IG_TP = float(c_p_IG_TP_expr.subs(subs_post))
        c_v_2_IG_TP = float(c_v_IG_TP_expr.subs(subs_post))
        c_p_2 = float(sp.re(c_p_expr.subs(subs_post)))
        c_v_2 = float(sp.re(c_v_expr.subs(subs_post)))
        h_2_IG_TP = float(h_IG_TP_expr.subs(subs_post))
        h_2 = float(sp.re(h_expr.subs(subs_post)))
        delta_h_2_IG_TP = float(sp.re(delta_h_IG_TP_expr.subs(subs_post)))
        delta_h_2_miller = float(delta_h_miller_expr.subs(subs_post))
        eqn_ns_1_soln_error = float(eqn_ns_1.subs(subs_post))
        eqn_ns_2_soln_error = float(sp.re(eqn_ns_2.subs(subs_post)))
        
        # -------------------------
        # SENSITIVITY ANALYSIS
        # -------------------------
        # Define vectors for v and T around the post-shock solution
        v_vec = np.linspace(v_2 * lower_lim_factor, v_2 * upper_lim_factor, num_intervals_per_var)
        T_vec = np.linspace(T_2 * lower_lim_factor, T_2 * upper_lim_factor, num_intervals_per_var)
        V_VEC, T_VEC = np.meshgrid(v_vec, T_vec)
        
        # Preallocate arrays for sensitivity variables
        c_p_IG_vec = np.zeros_like(V_VEC)
        c_v_IG_vec = np.zeros_like(V_VEC)
        c_p_vec = np.zeros_like(V_VEC)
        c_v_vec = np.zeros_like(V_VEC)
        c_vec = np.zeros_like(V_VEC)
        h_IG_TP_vec = np.zeros_like(V_VEC)
        delta_h_IG_TP_vec = np.zeros_like(V_VEC)
        delta_h_miller_vec = np.zeros_like(V_VEC)
        h_vec = np.zeros_like(V_VEC)
        eqn_ns_1_vec = np.zeros_like(V_VEC)
        eqn_ns_2_vec = np.zeros_like(V_VEC)
        
        print("\tComputing sensitivity analysis of the solution to [v, T] -- this may take a few moments...")
        for i in range(len(v_vec)):
            for j in range(len(T_vec)):
                subs_sens = {v_sym: v_vec[i], T_sym: T_vec[j]}
                c_p_IG_vec[j, i] = float(c_p_IG_TP_expr.subs(subs_sens))
                c_v_IG_vec[j, i] = float(c_v_IG_TP_expr.subs(subs_sens))
                c_p_vec[j, i] = float(sp.re(c_p_expr.subs(subs_sens)))
                c_v_vec[j, i] = float(sp.re(c_v_expr.subs(subs_sens)))
                c_vec[j, i] = float(sp.re(c_expr.subs(subs_sens)))
                h_IG_TP_vec[j, i] = float(h_IG_TP_expr.subs(subs_sens))
                delta_h_IG_TP_vec[j, i] = float(sp.re(delta_h_IG_TP_expr.subs(subs_sens)))
                delta_h_miller_vec[j, i] = float(sp.re(delta_h_miller_expr.subs(subs_sens)))
                h_vec[j, i] = float(sp.re(h_expr.subs(subs_sens)))
                eqn_ns_1_vec[j, i] = float(eqn_ns_1.subs(subs_sens))
                eqn_ns_2_vec[j, i] = float(sp.re(eqn_ns_2.subs(subs_sens)))
            print(f"\t\tSensitivity analysis {100*(i+1)/len(v_vec):.0f}% complete...")
        
        # -------------------------
        # OUTPUTS - COMMAND WINDOW RESULTS
        # -------------------------
        print("\nNow outputting command window results...")
        print("=============================================================")
        print("PRE-SHOCK")
        print("=============================================================")
        print("\nThermodynamics:")
        print(f"\tp_1_IG = {p_1_IG:.6f}\t\tp_1 = {p_1:.6f} [Pa]")
        print(f"\tp_1_IG/p_c = {p_1_IG/p_c_val:.6f}\t\tp_1/p_c = {p_1/p_c_val:.6f}")
        print(f"\tv_1 = {v1:.6f} [m^3 kmol^-1] (rho_1 = {rho1:.6f} [kg m^-3])")
        print(f"\tT_1 = {T1:.6f} [K]")
        print(f"\tT_1/T_c = {T1/T_c_val:.6f}")
        print(f"\tu_1 = {u1:.6f} [m s^-1]")
        print(f"\tc_1_IG_CP = {c_1_IG_CP:.6f}\t\t\tc_1 = {c_1:.6f} [m s^-1]")
        print(f"\tMa_1_IG_CP = {Ma_1_IG_CP:.6f}\t\t\tMa_1 = {Ma_1:.6f}")
        print("\n\tc_p_1_IG_TP = {:.6f}\t\tc_p_1 = {:.6f} [J kmol^-1 K^-1]".format(c_p_1_IG_TP, c_p_1))
        print("\tc_v_1_IG_TP = {:.6f}\t\tc_v_1 = {:.6f} [J kmol^-1 K^-1]".format(c_v_1_IG_TP, c_v_1))
        print("\th_1_IG_TP = {:.6f}\t\th_1 = {:.6f} [J kmol^-1]".format(h_1_IG_TP, h_1))
        print(f"\tdelta_h_1_IG_TP = {delta_h_1_IG_TP:.6f} [J kmol^-1]")
        print(f"\tdelta_h_1_miller = {delta_h_1_miller:.6f} [J kmol^-1]")
        print("\nReal-gas derivatives and integrals:")
        print(f"\texp_dpdT_1 = {exp_dpdT_1:.6f} [Pa K^-1]")
        print(f"\texp_dpdv_1 = {exp_dpdv_1:.6f} [J kmol^-1]")
        print(f"\texp_d2pdT2 = {exp_d2pdT2_1:.6f} [Pa K^-2]")
        print(f"\texp_intdpdT_1 = {exp_intdpdT_1:.6f} [J kmol^-1 K^-1]")
        print(f"\texp_intvdpdv_1 = {exp_intvdpdv_1:.6f} [J kmol^-1]")
        print(f"\texp_intd2pdT2_1 = {exp_intd2pdT2_1:.6f} [J kmol^-1 K^-2]")
        print("=============================================================")
        print("POST-SHOCK")
        print("=============================================================")
        print("\nThermodynamics:")
        print(f"\tp_2_IG_CP = {p_2_IG_CP:.6f}\t\tp_2 = {p_2:.6f} [Pa]")
        print(f"\tp_2_IG_CP/p_c = {p_2_IG_CP/p_c_val:.6f}\t\tp_2/p_c = {p_2/p_c_val:.6f}")
        print(f"\tv_2_IG_CP = {v_2_IG_CP:.6f}\t\t\tv_2 = {v_2:.6f} [m^3 kmol^-1]")
        print(f"\tT_2_IG_CP = {T_2_IG_CP:.6f}\t\t\tT_2 = {T_2:.6f} [K]")
        print(f"\tT_2_IG_CP/T_c = {T_2_IG_CP/T_c_val:.6f}\t\tT_2/T_c = {T_2/T_c_val:.6f}")
        print(f"\tu_2_IG_CP = {u_2_IG_CP:.6f}\t\t\tu_2 = {u_2:.6f} [m s^-1]")
        print(f"\tc_2_IG_CP = {c_2_IG_CP:.6f}\t\t\tc_2 = {c_2:.6f} [m s^-1]")
        print(f"\tMa_2_IG_CP = {Ma_2_IG_CP:.6f}\t\t\tMa_2 = {Ma_2:.6f}")
        print("\n\tc_p_2_IG_TP = {:.6f}\t\tc_p_2 = {:.6f} [J kmol^-1 K^-1]".format(c_p_2_IG_TP, c_p_2))
        print("\tc_v_2_IG_TP = {:.6f}\t\tc_v_2 = {:.6f} [J kmol^-1 K^-1]".format(c_v_2_IG_TP, c_v_2))
        print("\th_2_IG_TP = {:.6f}\t\th_2 = {:.6f} [J kmol^-1]".format(h_2_IG_TP, h_2))
        print(f"\tdelta_h_2_IG_TP = {delta_h_2_IG_TP:.6f} [J kmol^-1]")
        print(f"\tdelta_h_2_miller = {delta_h_2_miller:.6f} [J kmol^-1]")
        print("=============================================================")
        print("RATIOS OF P, V, T, u")
        print("=============================================================")
        print(f"\tp2p1_IG_CP = {p2p1_IG_CP:.6f}\t\t\tp2p1 = {p2p1:.6f}")
        print(f"\tv2v1_IG_CP = {v2v1_IG_CP:.6f}\t\t\tv2v1 = {v2v1:.6f}")
        print(f"\tT2T1_IG_CP = {T2T1_IG_CP:.6f}\t\t\tT2T1 = {T2T1:.6f}")
        print(f"\tu2u1_IG_CP = {u2u1_IG_CP:.6f}\t\t\tu2u1 = {u2u1:.6f}")
        
        # Optionally store the results for this (eos, run) case
        results[(eos, run_idx)] = {
            'p_1_IG': p_1_IG, 'p_1': p_1, 'v1': v1, 'T1': T1, 'u1': u1,
            'c_1_IG_CP': c_1_IG_CP, 'c_1': c_1, 'Ma_1_IG_CP': Ma_1_IG_CP, 'Ma_1': Ma_1,
            'h_1_IG_TP': h_1_IG_TP, 'h_1': h_1,
            'p_2_IG_CP': p_2_IG_CP, 'p_2': p_2, 'v_2_IG_CP': v_2_IG_CP, 'v_2': v_2,
            'T_2_IG_CP': T_2_IG_CP, 'T_2': T_2, 'u_2_IG_CP': u_2_IG_CP, 'u_2': u_2,
            'c_2_IG_CP': c_2_IG_CP, 'c_2': c_2, 'Ma_2_IG_CP': Ma_2_IG_CP, 'Ma_2': Ma_2,
            'p2p1_IG_CP': p2p1_IG_CP, 'p2p1': p2p1,
            'v2v1_IG_CP': v2v1_IG_CP, 'v2v1': v2v1,
            'T2T1_IG_CP': T2T1_IG_CP, 'T2T1': T2T1,
            'u2u1_IG_CP': u2u1_IG_CP, 'u2u1': u2u1,
            'Ma2Ma1_IG_CP': Ma2Ma1_IG_CP, 'Ma2Ma1': Ma2Ma1,
            'eqn_ns_1_error': eqn_ns_1_soln_error, 'eqn_ns_2_error': eqn_ns_2_soln_error
        }
        
        # -------------------------
        # SENSITIVITY ANALYSIS GRAPHS
        # -------------------------
        print("\nNow plotting sensitivity analysis graphs...")
        if show_graph_sensitivity_cpcv:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(V_VEC, T_VEC, c_p_IG_vec, color=[1, 0, 0], edgecolor='black')
            ax.plot_surface(V_VEC, T_VEC, c_v_IG_vec, color=[0, 0, 1], edgecolor='black')
            ax.plot_surface(V_VEC, T_VEC, c_p_vec, color=[1, 150/255, 0], edgecolor='black')
            ax.plot_surface(V_VEC, T_VEC, c_v_vec, color=[0, 150/255, 1], edgecolor='black')
            ax.set_xlabel('v [m^3 kmol^-1]')
            ax.set_ylabel('T [K]')
            ax.set_zlabel('[J kmol^-1 K^-1]')
            ax.legend(['c\'_p', 'c\'_v', 'c_p', 'c_v'])
            ax.set_title('Specific Heat Capacities')
            plt.show()
        if show_graph_sensitivity_c:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(V_VEC, T_VEC, c_vec)
            ax.set_xlabel('v [m^3 kmol^-1]')
            ax.set_ylabel('T [K]')
            ax.set_zlabel('[m s^-1]')
            ax.set_title('Real-Gas Speed of Sound')
            plt.show()
        if show_graph_sensitivity_h_IG:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(V_VEC, T_VEC, h_IG_TP_vec)
            ax.set_xlabel('v [m^3 kmol^-1]')
            ax.set_ylabel('T [K]')
            ax.set_zlabel('[J kmol^-1]')
            ax.set_title('Ideal Gas Enthalpy')
            plt.show()
        if show_graph_sensitivity_deltah:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(V_VEC, T_VEC, delta_h_IG_TP_vec, color='red', edgecolor='black')
            ax.plot_surface(V_VEC, T_VEC, delta_h_miller_vec, color='blue', edgecolor='black')
            ax.set_xlabel('v [m^3 kmol^-1]')
            ax.set_ylabel('T [K]')
            ax.set_zlabel('[J kmol^-1]')
            ax.legend(['Δh\'', 'Δh\'_{miller}'])
            ax.set_title('Enthalpy Departure Functions')
            plt.show()
        if show_graph_sensitivity_h:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(V_VEC, T_VEC, h_vec)
            ax.set_xlabel('v [m^3 kmol^-1]')
            ax.set_ylabel('T [K]')
            ax.set_zlabel('[J kmol^-1]')
            ax.set_title('Real Gas Enthalpy')
            plt.show()
        if show_graph_sensitivity_goveqn and run_idx == index_TRAC:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot_surface(V_VEC, T_VEC, eqn_ns_1_vec, color='red', edgecolor='black')
            ax1.scatter(v_2, T_2, eqn_ns_1_soln_error, s=150, c=[(0, 1, 1)])
            ax1.set_xlabel('v_2')
            ax1.set_ylabel('T_2')
            ax1.set_zlabel('Continuity-Momentum')
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_surface(V_VEC, T_VEC, eqn_ns_2_vec, color='blue', edgecolor='black')
            ax2.scatter(v_2, T_2, eqn_ns_2_soln_error, s=150, c=[(50/255, 200/255, 50/255)])
            ax2.set_xlabel('v_2')
            ax2.set_ylabel('T_2')
            ax2.set_zlabel('Continuity-Energy')
            plt.suptitle('Governing Equations Error')
            plt.show()
        
        print("Sensitivity analysis graphs complete.\n")
        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------\n")
        
# -------------------------
# FINAL BATCH ANALYSIS GRAPHS
# -------------------------
print("\nNow displaying final graphs...")
if show_graphs_final:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # These final plots are placeholders; you can aggregate the 'results' as needed.
    axs[0, 0].plot([0, 1], [1, 2], '-', color=color_ideal)
    axs[0, 0].set_ylabel('p_2/p_1')
    axs[0, 0].set_title('Pressure Ratio')
    axs[0, 0].grid(True)
    
    axs[0, 1].plot([0, 1], [0.2, 0.5], '-', color=color_RK)
    axs[0, 1].set_ylabel('v_2/v_1')
    axs[0, 1].set_title('Volume Ratio')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot([0, 1], [1, 2.25], '-', color=color_SRK)
    axs[1, 0].set_xlabel('Ma_1')
    axs[1, 0].set_ylabel('T_2/T_1')
    axs[1, 0].set_title('Temperature Ratio')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot([0, 1], [0.1, 0.4], '-', color=color_PR)
    axs[1, 1].set_xlabel('Ma_1')
    axs[1, 1].set_ylabel('Ma_2/Ma_1')
    axs[1, 1].set_title('Mach Number Ratio')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.title('Trust-Region-Dogleg Error')
    plt.xlabel('Ma_1')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

print("Program execution completed.")
