# analytics.py
import numpy as np
from scipy.optimize import fsolve
from abc import ABC, abstractmethod
from scipy.constants import R

class EOS(ABC):
    """Abstract Base Class for Equations of State"""
    
    @abstractmethod
    def compute_properties(self, T: float, P: float) -> tuple:
        """
        Compute thermodynamic properties from T and P
        Returns:
            density (kg/m³), enthalpy (J/kg), speed_of_sound (m/s)
        """
        pass
    
    @abstractmethod
    def required_params(self) -> list:
        """List of required parameters for EOS initialization"""
        pass


# ---------------------------------------------------------------------------
# 1) Van der Waals (VDW)
# ---------------------------------------------------------------------------
class VanDerWaalsEOS(EOS):
    """
    Van der Waals EOS:
      p = R*T / (v - b) - a / v^2,
    with
      a = 27 R^2 * Tc^2 / (64 Pc),
      b = R * Tc / (8 Pc).
    We'll do a numeric bracket to solve for v, then compute density = Mw / v.
    Enthalpy and speed_of_sound are placeholders.
    """
    def __init__(self, Tc: float, Pc: float, Mw: float):
        self.Tc = Tc
        self.Pc = Pc
        self.Mw = Mw
        # Gas constant for mixture/fluid:
        # R is universal in J/(mol*K), so we do fluid R_g = R / (Mw in kg/mol) for "J/(kg*K)"
        self.Rg = R / Mw
        # VdW a,b from critical props
        self.a = 27.0 * (self.Rg**2) * (Tc**2) / (64.0*Pc)
        self.b = (self.Rg * Tc)/(8.0 * Pc)

    def required_params(self):
        return ['Tc', 'Pc', 'Mw']
    
    def compute_properties(self, T: float, P: float) -> tuple:
        # bracket to solve p_calc - P = 0
        def vdw_resid(vol):
            if vol <= self.b:
                return 1e12
            p_calc = (self.Rg * T)/(vol - self.b) - self.a/(vol*vol)
            return p_calc - P
        
        v_low, v_high = 1e-6, 1.0
        for _ in range(200):
            v_mid = 0.5*(v_low + v_high)
            f_low = vdw_resid(v_low)
            f_mid = vdw_resid(v_mid)
            if f_low*f_mid <= 0:
                v_high = v_mid
            else:
                v_low = v_mid
            if abs(f_mid) < 1e-7:
                break
        v_sol = 0.5*(v_low + v_high)

        density = self.Mw / v_sol  # Mw [kg/mol], v_sol [m^3/kg], so overall [kg/m^3]

        # Simplistic placeholders for enthalpy + speed of sound
        cp_approx = 1500.0  # J/(kg*K)
        enthalpy = cp_approx * T
        gamma_approx = 1.3
        c = np.sqrt(gamma_approx * self.Rg * T)
        return density, enthalpy, c


# ---------------------------------------------------------------------------
# 2) Redlich–Kwong (RK)
# ---------------------------------------------------------------------------
class RedlichKwongEOS(EOS):
    """
    RK:
      p = R*T/(v - b) - a/(v(v+b)*sqrt(T)),
    with
      a=0.42748*(R^2 *Tc^{2.5})/Pc,
      b=0.08664*(R *Tc)/Pc
    """
    def __init__(self, Tc: float, Pc: float, Mw: float):
        self.Tc = Tc
        self.Pc = Pc
        self.Mw = Mw
        self.Rg = R / Mw
        self.a = 0.42748 * (self.Rg**2) * (Tc**2.5) / Pc
        self.b = 0.08664 * (self.Rg * Tc) / Pc

    def required_params(self):
        return ['Tc', 'Pc', 'Mw']

    def compute_properties(self, T: float, P: float) -> tuple:
        def rk_res(vol):
            if vol <= self.b:
                return 1e12
            p_calc = (self.Rg*T)/(vol - self.b) - self.a/(vol*(vol + self.b)*np.sqrt(T))
            return p_calc - P

        v_low, v_high = 1e-6, 1.0
        for _ in range(200):
            v_mid = 0.5*(v_low + v_high)
            f_low = rk_res(v_low)
            f_mid = rk_res(v_mid)
            if f_low*f_mid <= 0:
                v_high = v_mid
            else:
                v_low = v_mid
            if abs(f_mid) < 1e-7:
                break
        v_sol = 0.5*(v_low + v_high)

        density = self.Mw / v_sol
        cp_approx = 1500.0
        enthalpy = cp_approx * T
        gamma_approx = 1.3
        c = np.sqrt(gamma_approx * self.Rg * T)
        return density, enthalpy, c


# ---------------------------------------------------------------------------
# 3) Soave–Redlich–Kwong (SRK)
# ---------------------------------------------------------------------------
class SoaveRedlichKwongEOS(EOS):
    """
    SRK modifies alpha(T) in the RK form with:
      alpha(T) = [1 + k(1 - sqrt(T/Tc))]^2
      p = R*T/(v - b) - a*alpha / [v(v + b)]
    with
      a=0.42748(R^2 Tc^2/Pc), b=0.08664(R Tc/Pc), k=0.48 + 1.574*omega -0.176*omega^2
    """
    def __init__(self, Tc: float, Pc: float, Mw: float, omega: float):
        self.Tc = Tc
        self.Pc = Pc
        self.Mw = Mw
        self.omega = omega
        self.Rg = R / Mw

        self.a0 = 0.42748 * (self.Rg**2) * (Tc**2) / Pc
        self.b = 0.08664 * (self.Rg * Tc) / Pc
        self.k = 0.48 + 1.574*omega - 0.176*omega*omega

    def required_params(self):
        return ['Tc', 'Pc', 'Mw', 'omega']

    def alpha(self, T: float) -> float:
        Tr = T / self.Tc
        return (1 + self.k*(1 - np.sqrt(Tr)))**2

    def compute_properties(self, T: float, P: float) -> tuple:
        a_val = self.a0 * self.alpha(T)
        b = self.b
        
        def srk_res(vol):
            if vol <= b:
                return 1e12
            p_calc = (self.Rg*T)/(vol - b) - a_val/(vol*(vol + b))
            return p_calc - P

        v_low, v_high = 1e-6, 1.0
        for _ in range(200):
            v_mid = 0.5*(v_low + v_high)
            f_low = srk_res(v_low)
            f_mid = srk_res(v_mid)
            if f_low*f_mid <= 0:
                v_high = v_mid
            else:
                v_low = v_mid
            if abs(f_mid) < 1e-7:
                break

        v_sol = 0.5*(v_low + v_high)
        density = self.Mw / v_sol

        cp_approx = 1500.0
        enthalpy = cp_approx * T
        gamma_approx = 1.3
        c = np.sqrt(gamma_approx * self.Rg * T)
        return density, enthalpy, c


# ---------------------------------------------------------------------------
# 4) Peng–Robinson (PR) (unchanged from your original, just for completeness)
# ---------------------------------------------------------------------------
class PengRobinsonEOS(EOS):
    """Peng-Robinson Equation of State Implementation"""
    
    def __init__(self, Tc: float, Pc: float, Mw: float, omega: float = 0.344):
        self.Tc = Tc  # Critical temperature (K)
        self.Pc = Pc  # Critical pressure (Pa)
        self.Mw = Mw  # Molecular weight (kg/mol)
        self.omega = omega  # Acentric factor
        
        # PR constants
        # R is universal in J/(mol*K), so no / Mw here. We'll handle density with Z in final step
        self.a = 0.45724 * (R**2 * self.Tc**2) / self.Pc
        self.b = 0.07780 * R * self.Tc / self.Pc
        
    def required_params(self):
        return ['Tc', 'Pc', 'Mw', 'omega']
    
    def alpha(self, Tr: float) -> float:
        """Temperature-dependent alpha function"""
        return (1 + (0.37464 + 1.54226*self.omega - 0.26992*self.omega**2)
                * (1 - np.sqrt(Tr)))**2

    def compute_properties(self, T: float, P: float) -> tuple:
        A = self.a * self.alpha(T/self.Tc) * P / (R*T)**2
        B = self.b * P / (R*T)
        
        # Solve cubic equation for Z
        coeffs = [1,
                  -(1 - B),
                  (A - 2*B - 3*B*B),
                  -(A*B - B*B - B*B*B)]
        roots = np.roots(coeffs)
        # take the largest real root
        Z = max(roots[np.isreal(roots)].real)
        
        # density in kg/m^3
        # P * Mw / (Z * R * T), because R is J/(mol*K) and Mw is kg/mol
        density = P * self.Mw / (Z * R * T)
        
        # Enthalpy calculation (departure-based, simplified)
        h_departure = R*T*(Z - 1) - (
            self.a * self.alpha(T/self.Tc)
            * np.log((Z + (1+np.sqrt(2))*B)/(Z + (1-np.sqrt(2))*B))
          )/(2*np.sqrt(2)*self.b)
        enthalpy = h_departure / self.Mw  # J/kg
        
        # Speed of sound (simplified)
        c = np.sqrt(R*T/self.Mw * (1 + (self.a * self.alpha(T/self.Tc) * P)
                                / (R*T)**2 * (1 + np.sqrt(2))**2))
        return density, enthalpy, c


# ---------------------------------------------------------------------------
# 5) Span–Wagner (SW) placeholder
# ---------------------------------------------------------------------------
class SpanWagnerEOS(EOS):
    """Span-Wagner EOS Implementation (Skeleton)"""
    def __init__(self, coefficients: dict):
        # Should implement actual polynomials, multi-params, etc.
        self.coeffs = coefficients
        
    def required_params(self):
        return ['coefficients']  # Actual implementation needs specific SW parameters
    
    def compute_properties(self, T: float, P: float) -> tuple:
        # Not implemented
        raise NotImplementedError("Span-Wagner requires detailed input.")


# ---------------------------------------------------------------------------
# 6) Virial (truncated) placeholder
# ---------------------------------------------------------------------------
class VirialEOS(EOS):
    """
    p = rho * R_g * T ( 1 + B2(T)*rho ), ignoring higher terms.
    We'll do a bracket to solve for rho. 
    """
    def __init__(self, Mw: float, b0: float = 1e-3, b1: float = 1.0):
        self.Mw = Mw
        self.Rg = R / Mw
        self.b0 = b0
        self.b1 = b1
    
    def required_params(self):
        return ['Mw']  # plus b0, b1 if you want them as well
    
    def B2(self, T: float) -> float:
        return self.b0 - self.b1/T
    
    def compute_properties(self, T: float, P: float) -> tuple:
        def virial_res(rho):
            # p_calc = rho * Rg * T (1 + B2*rho)
            return rho*self.Rg*T*(1 + self.B2(T)*rho) - P
        
        rho_low, rho_high = 1e-6, 1e3
        for _ in range(200):
            mid = 0.5*(rho_low + rho_high)
            f_low = virial_res(rho_low)
            f_mid = virial_res(mid)
            if f_low*f_mid <= 0:
                rho_high = mid
            else:
                rho_low = mid
            if abs(f_mid) < 1e-7:
                break
        
        rho_sol = 0.5*(rho_low + rho_high)
        # approximate enthalpy
        cp_approx = 1500.0
        enthalpy = cp_approx*T
        gamma_approx = 1.3
        c = np.sqrt(gamma_approx*self.Rg*T)
        return rho_sol, enthalpy, c


# ---------------------------------------------------------------------------
# 7) IdealGasEOS
# ---------------------------------------------------------------------------
class IdealGasEOS(EOS):
    """Ideal Gas Equation of State"""
    
    def __init__(self, gamma: float = 1.4, Mw: float = 0.02897):
        self.gamma = gamma
        self.Mw = Mw
        self.Rg = R / Mw  # J/(kg*K)
        
    def required_params(self):
        return ['gamma', 'Mw']
    
    def compute_properties(self, T: float, P: float) -> tuple:
        density = P / (self.Rg * T)
        enthalpy = self.gamma/(self.gamma - 1.0) * self.Rg * T
        c = np.sqrt(self.gamma * self.Rg * T)
        return density, enthalpy, c


class ShockAnalyzer:
    """Generic shock wave analyzer for any EOS"""
    
    def __init__(self, eos: EOS):
        self.eos = eos
        
    def rankine_hugoniot(self, pre_state: tuple) -> tuple:
        """
        Solve Rankine-Hugoniot equations for given EOS
        Args:
            pre_state: (P1, T1, u1) in (Pa, K, m/s)
        Returns:
            (P2, T2, u2, rho2) post-shock state
        """
        P1, T1, u1 = pre_state
        
        # Pre-shock
        rho1, h1, c1 = self.eos.compute_properties(T1, P1)
        
        def equations(vars_):
            P2, T2 = vars_
            rho2, h2, c2 = self.eos.compute_properties(T2, P2)
            
            # Let's define:
            # - momentum eq (mom_eq)
            # - energy eq (energy_eq)
            # We'll do a simplified approach as an example. 
            
            # Mass eq: rho1 u1 = rho2 u2 => u2 = (rho1 u1) / rho2
            u2_ = (rho1*u1)/rho2
            
            # Momentum eq: P1 + rho1*u1^2 = P2 + rho2*u2^2
            mom_eq = (P2 - P1) + (rho2*u2_**2 - rho1*u1**2)
            
            # Energy eq (Bernoulli/h enthalpy form):
            # h1 + 0.5*u1^2 = h2 + 0.5*u2^2
            energy_eq = (h2 + 0.5*u2_**2) - (h1 + 0.5*u1**2)
            
            return [mom_eq, energy_eq]
        
        # We'll guess P2 ~ 2*P1, T2 ~ 1.2*T1 for example, or any naive approach
        guess = [2*P1, 1.2*T1]
        sol = fsolve(equations, guess, xtol=1e-6)
        P2, T2 = sol
        rho2, h2, c2 = self.eos.compute_properties(T2, P2)
        u2 = (rho1*u1)/rho2
        
        return P2, T2, u2, rho2

    def perturbation_correction(self, pre_state: tuple, epsilon=0.1) -> float:
        """Example of a weak-shock perturbation approach (stub)."""
        P1, T1, u1 = pre_state
        rho1, h1, c1 = self.eos.compute_properties(T1, P1)
        # do some trivial correction
        return P1*(1 + epsilon*(u1 / c1))


def create_eos(eos_type: str, params: dict) -> EOS:
    """
    Factory function for creating EOS instances
    with user-provided parameters, e.g.:
    
    create_eos('PR', {'Tc': 304.13, 'Pc': 7.3773e6, 'Mw': 0.04401, 'omega': 0.228})
    create_eos('ideal', {'gamma':1.4, 'Mw':0.02897})
    create_eos('VDW', {'Tc':304.13, 'Pc':7.3773e6, 'Mw':0.04401})
    ...
    """
    implementations = {
        'VDW': VanDerWaalsEOS,
        'RK': RedlichKwongEOS,
        'SRK': SoaveRedlichKwongEOS,
        'PR': PengRobinsonEOS,
        'SW': SpanWagnerEOS,
        'VIRIAL': VirialEOS,
        'IDEAL': IdealGasEOS
    }
    
    key = eos_type.upper()
    if key not in implementations:
        raise ValueError(f"Unsupported EOS type: {eos_type}. "
                         f"Must be one of {list(implementations.keys())}")
    cls = implementations[key]
    
    missing = [p for p in cls.required_params(cls) if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters for {eos_type}: {missing}")
    
    return cls(**params)

# Example Usage
if __name__ == "__main__":
    # Example: CO2, Peng-Robinson
    co2_params = {'Tc':304.13, 'Pc':7.3773e6, 'Mw':0.04401, 'omega':0.228}
    pr_eos = create_eos('PR', co2_params)
    analyzer = ShockAnalyzer(pr_eos)
    
    pre_state = (1e6, 300, 100)  # 1 MPa, 300 K, 100 m/s
    post_state = analyzer.rankine_hugoniot(pre_state)
    print("Peng-Robinson CO2 shock solution:")
    print(f"P2= {post_state[0]/1e6:.2f} MPa, T2= {post_state[1]:.1f} K, "
          f"u2= {post_state[2]:.2f} m/s, rho2= {post_state[3]:.3f} kg/m^3")

    # Example: Ideal gas (air)
    air_ideal = create_eos('ideal', {'gamma':1.4, 'Mw':0.02897})
    ideal_analyzer = ShockAnalyzer(air_ideal)
    post_ideal = ideal_analyzer.rankine_hugoniot(pre_state)
    print("\nIdeal Gas Air shock solution:")
    print(f"P2= {post_ideal[0]/1e6:.2f} MPa, T2= {post_ideal[1]:.1f} K, "
          f"u2= {post_ideal[2]:.2f} m/s, rho2= {post_ideal[3]:.3f} kg/m^3")

    # Example: VDW
    vdw_eos = create_eos('VDW', {'Tc':304.13, 'Pc':7.3773e6, 'Mw':0.04401})
    vdw_analyzer = ShockAnalyzer(vdw_eos)
    post_vdw = vdw_analyzer.rankine_hugoniot(pre_state)
    print("\nVan der Waals CO2 shock solution:")
    print(f"P2= {post_vdw[0]/1e6:.2f} MPa, T2= {post_vdw[1]:.1f} K, "
          f"u2= {post_vdw[2]:.2f} m/s, rho2= {post_vdw[3]:.3f} kg/m^3")
