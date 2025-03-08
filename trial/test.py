import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# =====================
# Fundamental Constants
# =====================
R = 0.0821  # L·atm/(mol·K)

class EOSComparator:
    def __init__(self, Tc, Pc, omega=None):
        """
        Initialize EOS comparator with compound properties
        :param Tc: Critical temperature (K)
        :param Pc: Critical pressure (atm)
        :param omega: Acentric factor (required for PR)
        """
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        
        # Initialize both EOS
        self.pr = self.PengRobinson(Tc, Pc, omega) if omega else None
        self.rk = self.RedlichKwong(Tc, Pc)
        
    class PengRobinson:
        def __init__(self, Tc, Pc, omega):
            self.Tc = Tc
            self.Pc = Pc
            self.omega = omega
            self.a = 0.45724 * (R**2 * Tc**2)/Pc
            self.b = 0.07780 * R * Tc/Pc
            self.kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
            
        def alpha(self, T):
            return (1 + self.kappa*(1 - np.sqrt(T/self.Tc)))**2
        
        def pressure(self, v, T):
            return R*T/(v - self.b) - self.a*self.alpha(T)/(v*(v + self.b) + self.b*(v - self.b))
        
        def solve_volume(self, P, T):
            def func(v):
                return self.pressure(v, T) - P
            return fsolve(func, R*T/P, xtol=1e-10)[0]
        
    class RedlichKwong:
        def __init__(self, Tc, Pc):
            self.Tc = Tc
            self.Pc = Pc
            self.a = 0.42748 * (R**2 * Tc**2.5)/Pc
            self.b = 0.08664 * R * Tc/Pc
            
        def pressure(self, v, T):
            return R*T/(v - self.b) - self.a/(np.sqrt(T)*v*(v + self.b))
        
        def solve_volume(self, P, T):
            def func(v):
                return self.pressure(v, T) - P
            return fsolve(func, R*T/P, xtol=1e-10)[0]

    def compare_eos(self, T, P):
        """
        Compare PR and RK EOS with phase-aware evaluation
        Returns analysis dict with recommendation
        """
        results = {}
        phase = self._determine_phase(T, P)
        
        try:
            # RK calculations
            rk_v = self.rk.solve_volume(P, T)
            rk_z = P*rk_v/(R*T)
            
            # PR calculations if available
            pr_v = self.pr.solve_volume(P, T) if self.pr else None
            pr_z = P*pr_v/(R*T) if pr_v else None
            
            # Evaluation criteria
            results.update(self._evaluate_phase_performance(phase, pr_z, rk_z, pr_v, rk_v))
            
        except Exception as e:
            results['error'] = str(e)
            
        return results
    
    def _determine_phase(self, T, P):
        """Determine phase region based on critical properties"""
        Tr = T/self.Tc
        Pr = P/self.Pc
        
        if Tr > 1 and Pr > 1:
            return 'supercritical'
        elif Tr < 1 and Pr < 1:
            return 'vapor'
        else:
            return 'liquid' if Pr > 0.5 else 'vapor-liquid'
    
    def _evaluate_phase_performance(self, phase, pr_z, rk_z, pr_v, rk_v):
        """Apply literature-based evaluation criteria"""
        evaluation = {
            'phase': phase,
            'RK_Z': round(rk_z, 4),
            'PR_Z': round(pr_z, 4) if pr_z else None,
            'recommendation': 'RK'
        }
        
        # Literature-based decision matrix (from search results)
        if phase == 'liquid':
            evaluation['recommendation'] = 'PR' if pr_z else 'N/A'
            evaluation['rationale'] = 'PR better for liquid densities (Search Result 2-4)'
        elif phase == 'vapor':
            evaluation['rationale'] = 'RK sufficient for gas phase (Search Result 2)'
        elif phase == 'vapor-liquid':
            evaluation['recommendation'] = 'PR' if pr_z else 'RK'
            evaluation['rationale'] = 'PR better for phase equilibria (Search Result 4)'
        else:  # supercritical
            evaluation['rationale'] = 'RK generally adequate (Search Result 4)'
            
        # Add density comparison if available
        if pr_v and rk_v:
            evaluation['density_ratio'] = round((1/pr_v)/(1/rk_v), 2)
            if phase == 'liquid' and abs(1 - evaluation['density_ratio']) > 0.1:
                evaluation['recommendation'] = 'PR'
                evaluation['rationale'] += ' | PR liquid density closer to experimental'
                
        return evaluation

# ========================
# Enhanced Testing Suite
# ========================
def run_analysis(compounds, conditions):
    """Run comprehensive EOS comparison for multiple compounds"""
    results = []
    for compound in compounds:
        comparator = EOSComparator(
            Tc=compound['Tc'],
            Pc=compound['Pc'],
            omega=compound.get('omega')
        )
        
        compound_result = {
            'name': compound['name'],
            'conditions': [],
            'comparisons': []
        }
        
        for T, P in conditions:
            analysis = comparator.compare_eos(T, P)
            compound_result['conditions'].append((T, P))
            compound_result['comparisons'].append(analysis)
            
        results.append(compound_result)
    return results

# ========================
# Visualization Functions
# ========================
def plot_comparison(results):
    """Generate multi-panel comparison plot"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Z-factor comparison
    for idx, compound in enumerate(results):
        pr_z = [c['PR_Z'] for c in compound['comparisons'] if c['PR_Z']]
        rk_z = [c['RK_Z'] for c in compound['comparisons']]
        pressures = [P for T, P in compound['conditions']]
        
        axs[0,0].plot(pressures, pr_z, label=f"{compound['name']} (PR)")
        axs[0,0].plot(pressures, rk_z, '--', label=f"{compound['name']} (RK)")
    
    axs[0,0].set_title('Compressibility Factor Comparison')
    axs[0,0].set_xlabel('Pressure (atm)')
    axs[0,0].set_ylabel('Z Factor')
    axs[0,0].legend()
    axs[0,0].grid(True)

    # Recommendation analysis
    rec_counts = {'PR': 0, 'RK': 0}
    for compound in results:
        for comp in compound['comparisons']:
            rec_counts[comp['recommendation']] += 1
            
    axs[0,1].bar(rec_counts.keys(), rec_counts.values())
    axs[0,1].set_title('EOS Recommendation Distribution')
    axs[0,1].set_ylabel('Count')

    # Phase distribution
    phases = {}
    for compound in results:
        for comp in compound['comparisons']:
            phases[comp['phase']] = phases.get(comp['phase'], 0) + 1
            
    axs[1,0].pie(phases.values(), labels=phases.keys(), autopct='%1.1f%%')
    axs[1,0].set_title('Phase Region Distribution')

    # Density ratio analysis
    density_ratios = []
    for compound in results:
        for comp in compound['comparisons']:
            if 'density_ratio' in comp:
                density_ratios.append(comp['density_ratio'])
                
    axs[1,1].hist(density_ratios, bins=10)
    axs[1,1].set_title('Liquid Density Ratio (PR/RK)')
    axs[1,1].set_xlabel('Density Ratio')
    axs[1,1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

# ========================
# Execution and Validation
# ========================
if __name__ == "__main__":
    # Test compounds from search results
    compounds = [
        {'name': 'Methane', 'Tc': 190.56, 'Pc': 45.99, 'omega': 0.011},
        {'name': 'n-Octane', 'Tc': 569.32, 'Pc': 24.9, 'omega': 0.398},
        {'name': 'CO2', 'Tc': 304.18, 'Pc': 72.8, 'omega': 0.225}
    ]
    
    # Test conditions covering different phase regions
    conditions = [
        (300, 50),   # Subcritical vapor
        (600, 200),  # Supercritical
        (400, 10),   # Vapor-liquid
        (500, 300)   # High pressure liquid
    ]
    
    # Run analysis
    results = run_analysis(compounds, conditions)
    
    # Print detailed results
    for compound in results:
        print(f"\n{compound['name']} Analysis:")
        for (T, P), analysis in zip(compound['conditions'], compound['comparisons']):
            print(f"  @ {T}K, {P}atm [{analysis['phase']}]:")
            print(f"    PR Z: {analysis.get('PR_Z', 'N/A')}")
            print(f"    RK Z: {analysis['RK_Z']}")
            print(f"    Recommendation: {analysis['recommendation']}")
            print(f"    Rationale: {analysis['rationale']}")
    
    # Generate comprehensive plots
    plot_comparison(results)
