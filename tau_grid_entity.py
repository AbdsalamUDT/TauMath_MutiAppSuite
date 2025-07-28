# tau_grid_entity.py
# TauGrid Symbolic Node – Based on UDT Principles by Abdulsalam Al-Mayahi
# UK Patent GB2511966.0 – Union Dipole Theory (UDP) and τ-Dynamic Systems

import math
import numpy as np
from collections import deque

class TauGridEntity:
    """
    Symbolic representation of grid components using Union Dipole Theory (UDT).
    
    Developed by: Abdulsalam Al-Mayahi
    Based on: Union Dipole Particle (UDP) model and τ-resonance internal dynamics.
    Patent Reference: GB2511966.0

    Each grid element operates as a symbolic τ-entity with internal phase, resonance,
    and symbolic energy, enabling predictive behavior and self-stabilization.
    """

    def __init__(self, name, node_type, base_resonance):
        """
        Initialize the grid entity.

        Parameters:
        - name (str): Identifier of the grid component
        - node_type (str): Type of component (generator, transformer, transmission, substation)
        - base_resonance (float): Natural frequency or symbolic resonance of the entity
        """
        self.name = name
        self.type = node_type
        self.base_resonance = base_resonance
        self.phase = 0.0  # Internal τ-phase
        self.energy = 0.0  # Symbolic energy (0-100 scale)
        self.resonance_history = deque(maxlen=1000)
        self.properties = {
            'stability_factor': 0.97,
            'resonance_tolerance': 0.03,
            'load_capacity': 100,
            'current_load': 0
        }

    def update_state(self, delta_time, grid_phase):
        """
        Update the symbolic state of the component based on grid resonance.

        Parameters:
        - delta_time (float): Elapsed simulation time
        - grid_phase (float): Current global τ-phase of the grid

        Returns:
        - current_resonance (float): Updated symbolic resonance signal
        """
        # Phase alignment with grid
        phase_diff = (grid_phase - self.phase) % (2 * math.pi)
        alignment = math.cos(phase_diff)

        # Update symbolic energy based on alignment
        energy_change = alignment * delta_time * 0.8
        self.energy = np.clip(self.energy + energy_change, 0, 100)

        # Advance phase based on base resonance and τ-alignment
        self.phase = (self.phase + self.base_resonance * delta_time * (1 + alignment * 0.3)) % (2 * math.pi)

        # Current resonance signal
        current_resonance = math.sin(self.phase) * (self.energy / 100)
        self.resonance_history.append(current_resonance)

        return current_resonance

    def predict_failure(self):
        """
        Predict potential grid instability or failure using τ-signature variance and symbolic trend.

        Returns:
        - failure_prob (float): Estimated probability of symbolic failure [0.0 to 1.0]
        - failure_type (str): Symbolic classification of the predicted failure
        """
        if len(self.resonance_history) < 10:
            return 0.0, "INSUFFICIENT_DATA"

        recent = list(self.resonance_history)[-10:]
        variance = np.var(recent)
        trend = np.polyfit(range(10), recent, 1)[0]

        failure_prob = min(1.0, (variance ** 1.5) * 8 + abs(trend) * 15)

        if variance > 0.2 and trend > 0.05:
            failure_type = "OVERLOAD_IMMINENT"
        elif variance < 0.05 and trend < -0.1:
            failure_type = "UNDERLOAD_DAMAGE"
        else:
            failure_type = "STABILITY_LOSS"

        return failure_prob, failure_type

    def adjust_resonance(self, adjustment):
        """
        Adjust the entity’s base resonance to stabilize symbolic flow.

        Parameters:
        - adjustment (float): Factor to modify the base resonance

        Returns:
        - new_base_resonance (float): Updated base resonance
        """
        self.base_resonance *= (1 + adjustment)
        return self.base_resonance
