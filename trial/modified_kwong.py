import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# Constants for real air (from the paper)
R = 287.22  # Gas constant for air (J/kg·K)
pc = 37.66e5  # Critical pressure (Pa)
Tc = 132.52  # Critical temperature (K)

# Redlich-Kwong coefficients for air
a = (0.4278 * R**2 * Tc**2.5) / pc
b = (0.0867 * R * Tc) / pc

# Ideal gas heat capacity coefficients for air (from the paper)
A = 1.0115846e3
B = -1.0183346e-1
C = 2.7676571e-4

# Function to calculate enthalpy using RK EOS
def enthalpy(T, v, pref, Tref):
    h_ideal = A * T + (B / 2) * T**2 + (C / 3) * T**3 - (A * Tref + (B / 2) * Tref**2 + (C / 3) * Tref**3)
    delta_h = (-3 * a / (2 * b * np.sqrt(T))) * np.log(v / (v + b)) + R * T - (R * T / (v - b) - a / (np.sqrt(T) * v * (v + b))) * v
    return h_ideal - delta_h

# Function to calculate pressure using RK EOS
def pressure(T, v):
    return (R * T) / (v - b) - a / (np.sqrt(T) * v * (v + b))


# Function to solve the modified energy equation for v2
def energy_equation(v2, T1, p1, v1, c1):
    Q = (v2 - b) / (3 * R) * (p1 + (c1**2 / v1**2) * (v1 - v2))
    cos_theta = (a * (b - v2)) / (2 * R * v2 * (v2 + b) * Q**1.5)
    T2 = (2 * np.sqrt(Q) * np.cos(np.arccos(cos_theta) / 3))**2 # Solve for T2 using the cubic equation (Equation 32 in the paper)
    p2 = pressure(T2, v2)
    h1 = enthalpy(T1, v1, pref=1e5, Tref=300)  # Reference pressure and temperature
    h2 = enthalpy(T2, v2, pref=1e5, Tref=300)
    return h2 - h1 - (c1**2 / 2) * (1 - (v2**2 / v1**2))

# Function to calculate downstream state variables
def calculate_downstream(T1, p1, v1, c1):
    # Solve for v2 using fsolve
    v2_guess = v1 / 2  # Initial guess for v2
    v2 = fsolve(energy_equation, v2_guess, args=(T1, p1, v1, c1))[0]
    
    # Calculate T2 using the cubic equation (Equation 32 in the paper)
    Q = (v2 - b) / (3 * R) * (p1 + (c1**2 / v1**2) * (v1 - v2))
    cos_theta = (a * (b - v2)) / (2 * R * v2 * (v2 + b) * Q**1.5)
    T2 = (2 * np.sqrt(Q) * np.cos(np.arccos(cos_theta) / 3))**2
    
    # Calculate p2 using RK EOS
    p2 = pressure(T2, v2)
    
    # Calculate downstream velocity c2
    c2 = c1 * v1 / v2
    
    return T2, p2, v2, c2


# Function to validate the model against benchmark data
def validate_model(benchmark_data):
    errors = []
    for data in benchmark_data:
        T1, p1, v1, c1, T2_benchmark, p2_benchmark, v2_benchmark, M2_benchmark = data
        
        # Calculate downstream state variables using the model
        T2_model, p2_model, v2_model, c2_model = calculate_downstream(T1, p1, v1, c1)
        
        # Calculate Mach number downstream
        M2_model = c2_model / np.sqrt(1.4 * R * T2_model)  # Assuming gamma = 1.4 for air
        
        # Calculate relative errors
        error_T2 = np.abs((T2_model - T2_benchmark) / T2_benchmark)
        error_p2 = np.abs((p2_model - p2_benchmark) / p2_benchmark)
        error_v2 = np.abs((v2_model - v2_benchmark) / v2_benchmark)
        error_M2 = np.abs((M2_model - M2_benchmark) / M2_benchmark)
        
        errors.append((error_T2, error_p2, error_v2, error_M2))
    
    return errors

# Example benchmark data (replace with actual data from shock tables or the paper)
benchmark_data = [
    # Format: (T1, p1, v1, c1, T2_benchmark, p2_benchmark, v2_benchmark, M2_benchmark)
    (300, 1e5, 0.8, 500, 400, 2e5, 0.4, 0.6),  # Example data point 1
    (350, 1.2e5, 0.7, 600, 450, 2.5e5, 0.35, 0.55),  # Example data point 2
]

# Validate the model
errors = validate_model(benchmark_data)

def plot_shock_ratios(p1, T1, M1_array, gamma=1.4):
    """
    Plot p2/p1, v2/v1, T2/T1, and M2 vs. M1 on a single chart.
    Similar to the style of Fig. 1 in the paper.
    
    Parameters:
      p1, T1: Upstream static pressure (Pa) and temperature (K)
      M1_array: 1D array of upstream Mach numbers to evaluate
      gamma: (optional) approximate ratio of specific heats for the
             upstream speed of sound calculation. Default 1.4 for air.
    """

    # Approximate upstream speed of sound
    a1_approx = np.sqrt(gamma * R * T1)

    # Approximate upstream specific volume (v1)
    # If you want a more accurate real-gas v1, solve p1 = pressure(T1, v1) for v1.
    # For demonstration, we do v1 ~ R*T1 / p1:
    v1_approx = R * T1 / p1

    # Prepare arrays to store results
    p2p1_list = []
    v2v1_list = []
    T2T1_list = []
    M2_list   = []

    # Loop over each M1 in the array
    for M1 in M1_array:
        c1 = M1 * a1_approx

        # Solve for downstream states
        T2, p2, v2, c2 = calculate_downstream(T1, p1, v1_approx, c1)

        # Compute dimensionless ratios
        p2p1_list.append(p2 / p1)
        v2v1_list.append(v2 / v1_approx)
        T2T1_list.append(T2 / T1)

        # Approximate a2 (again, a real-gas formula would be more precise)
        a2_approx = np.sqrt(gamma * R * T2)
        M2 = c2 / a2_approx
        M2_list.append(M2)

    # Now make a single figure with all four curves
    plt.figure(figsize=(6, 5))

    # Plot each ratio
    plt.plot(M1_array, p2p1_list, 'o-', label=r'$p_2/p_1$')
    plt.plot(M1_array, v2v1_list, 's-', label=r'$v_2/v_1$')
    plt.plot(M1_array, T2T1_list, '^-', label=r'$T_2/T_1$')
    plt.plot(M1_array, M2_list,  'x-', label=r'$M_2$')

    # Style the chart
    plt.xlabel(r'Mach number $M_1$')
    plt.ylabel('Dimensionless ratios')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title(f'Normal Shock Wave at p1={p1/1e5:.1f} bar, T1={T1:.0f} K')
    plt.tight_layout()
    plt.show()


p1_example = 10e5  # 10 bar in Pa
T1_example = 700.0

    # Mach numbers from 1 to 5
M1_vals = np.linspace(1.0, 5.0, 30)

    # Plot
plot_shock_ratios(p1_example, T1_example, M1_vals)


# Print validation results
for i, (error_T2, error_p2, error_v2, error_M2) in enumerate(errors):
    print(f"Data Point {i + 1}:")
    print(f"  Relative Error in T2: {error_T2 * 100:.2f}%")
    print(f"  Relative Error in p2: {error_p2 * 100:.2f}%")
    print(f"  Relative Error in v2: {error_v2 * 100:.2f}%")
    print(f"  Relative Error in M2: {error_M2 * 100:.2f}%")