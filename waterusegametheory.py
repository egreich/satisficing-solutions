### Model that takes some mean of plant traits and returns a Nash equilibrium Evolutionary Stable Strategies (ESS) distribution
### using game theory/replicator dynamics
### Attempts to model plant trait distributions as more than uncertainties
### uncertainty distrubutions --> spatial and temporal plant trait distributions

### TBD: Take CliMA posterior mean trait output and recreate distributions using replicator dynamics


import numpy as np
import matplotlib.pyplot as plt


# Constants
D = 1.6  # Diffusivity ratio H2O/CO2
Γ = 40   # CO2 compensation point (μmol/mol)
Ca = 400 # Atmospheric CO2 (μmol/mol)
VPD = 1.5 # Vapor pressure deficit (kPa)

# Photosynthetic parameters
Vcmax = 80   # μmol/m²/s
Jmax = 160   # μmol/m²/s
Rd = 2.0     # Dark respiration (μmol/m²/s)


# Define simple physiological approximate equations
def vulnerability_curve(P, P50, a):
    """Vulnerability curve: fraction of conductivity remaining using a sigmoidal function"""
    # Use a sigmoidal function
    # PLC = 1 / (1 + exp(-a*(P - P50))) where P is negative
    return 1 / (1 + np.exp(-a * (P - P50)))

def hydraulic_conductance(P, P50, a, K_max):
    """Current hydraulic conductance at water potential P"""
    return K_max * vulnerability_curve(P, P50, a)

def find_critical_P(P_soil, P50, a, K_max, P_crit_threshold=0.05):
    """Find critical water potential where E_crit occurs"""
    P = np.linspace(P_soil, -10, 1000)
    K_vals = hydraulic_conductance(P, P50, a, K_max)
    crit_idx = np.where(K_vals <= P_crit_threshold * K_max)[0]
    return P[crit_idx[0]] if len(crit_idx) > 0 else P[-1]

def E_to_P(E, P_soil, P50, a, K_max):
    """Convert transpiration rate E to leaf water potential P"""
    # Handle case where E is 0
    if E <= 0:
        return P_soil
    
    P_guess = P_soil - E / K_max
    tol = 1e-6
    max_iter = 100
    
    for i in range(max_iter):
        K_current = hydraulic_conductance(P_guess, P50, a, K_max)
        f = K_current * (P_soil - P_guess) - E
        if abs(f) < tol:
            return P_guess
        
        # Derivative of f with respect to P
        dKdP = K_max * (-a * np.exp(a * (P_guess - P50))) / \
               (1 + np.exp(a * (P_guess - P50)))**2
        dfdP = dKdP * (P_soil - P_guess) - K_current
        
        # Avoid division by zero
        if abs(dfdP) < tol:
            break
            
        P_guess -= f / dfdP
    
    return P_guess

def E_crit(P_soil, P50, a, K_max):
    """Critical transpiration rate from vulnerability curve"""
    P_crit = find_critical_P(P_soil, P50, a, K_max)
    crit_E = K_max * (P_soil - P_crit)
    return max(0, crit_E)  # Ensure non-negative

def farquhar_photosynthesis(Ci, Vcmax, Jmax, Rd):
    """Simplified Farquhar model for net photosynthesis"""
    Wc = Vcmax * (Ci - Γ) / (Ci + 245)
    Wj = Jmax * (Ci - Γ) / (4 * Ci + 8 * Γ)
    A = min(Wc, Wj) - Rd
    return max(0, A)

def E_to_Ci(E, Ca, VPD):
    """Convert E to Ci using Fick's law"""
    return Ca - 1.6 * D * E / (VPD * 0.001)

# Define game theory components
class WaterUseGame:
    def __init__(self, community_params):
        """
        community_params: SMAP-derived parameters for the entire grid cell
        (e.g., aggregate P50, K_max, etc.)
        """
        self.community_params = community_params
        self.strategies = np.linspace(0.01, 0.99, 50)  # Range of stomatal opening strategies (0-1 scale)
        
    def calculate_payoff(self, strategy, community_strategy):
        """
        Calculate carbon gain for a plant using 'strategy' when the community
        average strategy is 'community_strategy'
        """
        # Convert strategy to actual transpiration rate
        E = strategy * self.community_params['E_crit_max']
        
        # Calculate individual plant carbon gain
        Ci = E_to_Ci(E, self.community_params['Ca'], self.community_params['VPD'])
        A = farquhar_photosynthesis(Ci, self.community_params['Vcmax'], 
                                   self.community_params['Jmax'], self.community_params['Rd'])
        
        # Game theory element: payoff depends on what others are doing
        # If everyone is using water aggressively, the resource depletes faster
        water_penalty = np.exp(-self.community_params['competition_factor'] * community_strategy)
        
        # Hydraulic risk penalty
        P_leaf = E_to_P(E, self.community_params['P_soil'], 
                              self.community_params['P50'], self.community_params['a'], self.community_params['K_max'])
        hydraulic_risk = 1 - hydraulic_conductance(P_leaf, self.community_params['P50'], self.community_params['a'], 
                                                 self.community_params['K_max']) / self.community_params['K_max']
        
        # Final payoff: carbon gain adjusted by competition and risk
        payoff = A * water_penalty * (1 - hydraulic_risk)
        
        return payoff


# Find evolutionally stable strategies
# Nash equilibrium where no individual plant can improve its payoff by unilaterally changing its strategy
def find_ess(game):
    """
    Find Evolutionary Stable Strategies using replicator dynamics
    """
    # Initialize population with different strategies
    population = game.strategies.copy()
    payoff_matrix = np.zeros((len(population), len(population)))
    
    # Calculate payoff for each strategy against each possible community composition
    for i, strategy in enumerate(population):
        for j, community_strategy in enumerate(population):
            payoff_matrix[i, j] = game.calculate_payoff(strategy, community_strategy)
    
    # Replicator dynamics to find ESS
    n_strategies = len(population)
    population_fraction = np.ones(n_strategies) / n_strategies  # Start with uniform distribution
    
    for generation in range(1000):  # Run for many generations
        # Calculate average payoff for each strategy
        avg_payoff = payoff_matrix @ population_fraction
        
        # Calculate total average payoff
        total_avg_payoff = population_fraction @ avg_payoff
        
        # Replicator equation: strategies grow in proportion to their relative success
        new_fraction = population_fraction * (avg_payoff / total_avg_payoff)
        
        # Check for convergence
        if np.max(np.abs(new_fraction - population_fraction)) < 1e-6:
            break
            
        population_fraction = new_fraction
    
    # Find the dominant strategy(ies)
    ess_candidates = population[population_fraction > 0.1]  # Strategies that persist
    return ess_candidates, population_fraction

# Example of how SMAP data informs the game parameters
community_params = {
    'E_crit_max': 0.05,  # From SMAP-derived hydraulic constraints
    'P50': -2.8,         # Community-level vulnerability from SMAP
    'K_max': 0.025,      # Aggregate hydraulic conductivity
    'a': 2.0,           # Shape parameter for vulnerability curv
    'P_soil': -0.6,      # SMAP soil moisture data
    'competition_factor': 2.0,  # How strongly plants compete for water (tuned parameter)
    'Ca': 400,
    'VPD': 1.8,
    'Vcmax': 90,         # Community-level photosynthetic capacity
    'Jmax': 180,
    'Rd': 2.0
}

# Run the game theory analysis
game = WaterUseGame(community_params)
ess_strategies, strategy_distribution = find_ess(game)

print(f"Evolutionary Stable Strategies: {ess_strategies}")
print(f"Strategy Distribution: {strategy_distribution}")


# Create strategy values (0.01, 0.03, 0.05, ...)
strategy_values = np.linspace(0.01, 0.99, len(strategy_distribution))


# Create a plot of the strategy distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Full distribution (log scale to see the tiny values)
ax1.bar(strategy_values, strategy_distribution, width=0.015, alpha=0.7, color='steelblue')
ax1.set_yscale('log')  # Log scale to see the exponential decay
ax1.set_xlabel('Stomatal Opening Strategy')
ax1.set_ylabel('Population Fraction (log scale)')
ax1.set_title('Full Evolutionary Strategy Distribution\n(Log Scale)')
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Zoomed-in view of the dominant strategies (linear scale)
# Only show the first 5 strategies that have meaningful populations
visible_cutoff = 5
ax2.bar(strategy_values[:visible_cutoff], strategy_distribution[:visible_cutoff], 
        width=0.005, alpha=0.7, color='steelblue')
ax2.set_xlabel('Stomatal Opening Strategy')
ax2.set_ylabel('Population Fraction')
ax2.set_title('Dominant Strategies Only\n(Linear Scale)')
ax2.grid(True, alpha=0.3)

# Add value labels on the bars for the zoomed-in plot
for i, (x, y) in enumerate(zip(strategy_values[:visible_cutoff], strategy_distribution[:visible_cutoff])):
    if y > 0.001:  # Only label meaningful values
        ax2.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Print a clean summary
print("=== EVOLUTIONARY STRATEGY DISTRIBUTION SUMMARY ===")
print(f"Total strategies simulated: {len(strategy_distribution)}")
print(f"Strategies with meaningful populations: {np.sum(strategy_distribution > 0.001)}")
print("\nDominant strategies:")
for i, (strategy, fraction) in enumerate(zip(strategy_values, strategy_distribution)):
    if fraction > 0.001:
        print(f"Strategy {i}: {strategy:.3f} → {fraction:.3%} of population")
    elif i < 10:  # Show first few even if tiny
        print(f"Strategy {i}: {strategy:.3f} → {fraction:.2e} (effectively extinct)")

# Calculate effective community behavior
dominant_mask = strategy_distribution > 0.001
effective_strategy = np.sum(strategy_values[dominant_mask] * strategy_distribution[dominant_mask])
print(f"\nEffective community strategy: {effective_strategy:.4f}")
print("This is the 'average' stomatal behavior emerging from competition")


###         Rational:
#           Optimization	                Evolutionary Game Theory
#Question	"What's the best strategy?"	    "What strategies can persist together?"
#Approach	Maximizes a function	        Simulates population dynamics
#Output	    Single best answer	            Stable distribution of strategies
#Biology	Individual rationality	        Population-level emergence