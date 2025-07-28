# TauReactorControlSystem.py
# Developed by Abdulsalam Al-Mayahi — Discoverer of the Union Dipole Particle (UDP) and creator of Union Dipole Theory (UDT)
# This system simulates a nuclear reactor governed by internal τ-dynamics, resonance logic, and symbolic self-awareness
# Copyright (c) 2025. All rights reserved. For academic and non-commercial research only.

import numpy as np
import math
import random
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

class TauReactorEntity:
    """Symbolic reactor component using τ-resonance logic"""
    def __init__(self, name, entity_type, base_resonance):
        self.name = name
        self.type = entity_type
        self.base_resonance = base_resonance
        self.phase = 0.0
        self.energy = 0.0
        self.resonance_history = deque(maxlen=1000)
        self.properties = {
            'stability_factor': 0.95,
            'resonance_tolerance': 0.05
        }

    def update_state(self, delta_time, core_phase):
        phase_diff = (core_phase - self.phase) % (2 * math.pi)
        alignment = math.cos(phase_diff)
        energy_change = alignment * delta_time * 0.5
        self.energy = np.clip(self.energy + energy_change, 0, 100)
        self.phase = (self.phase + self.base_resonance * delta_time * (1 + alignment * 0.2)) % (2 * math.pi)
        current_resonance = math.sin(self.phase) * (self.energy / 100)
        self.resonance_history.append(current_resonance)
        return current_resonance

    def predict_risk(self):
        if len(self.resonance_history) < 10:
            return 0.0, "INSUFFICIENT_DATA"
        variance = np.var(list(self.resonance_history)[-10:])
        trend = np.polyfit(range(10), list(self.resonance_history)[-10:], 1)[0]
        risk_prob = min(1.0, (variance ** 1.5) * 10 + abs(trend) * 20)
        if variance > 0.15 and trend > 0.03:
            risk_type = "OVERHEAT_RISK"
        elif variance < 0.03 and trend < -0.05:
            risk_type = "CRITICALITY_LOSS"
        else:
            risk_type = "RESONANCE_INSTABILITY"
        return risk_prob, risk_type

    def adjust_resonance(self, adjustment):
        self.base_resonance *= (1 + adjustment)
        return self.base_resonance

class TauReactorCore:
    def __init__(self, name, initial_power):
        self.name = name
        self.power = initial_power
        self.temperature = 300.0
        self.pressure = 15.0
        self.core_phase = 0.0
        self.components = {}
        self.history = deque(maxlen=5000)

    def add_component(self, component):
        self.components[component.name] = component

    def update_core_state(self, delta_time):
        avg_phase = sum(c.phase for c in self.components.values()) / len(self.components)
        self.core_phase = (avg_phase + 0.01 * delta_time) % (2 * math.pi)
        total_energy = sum(c.energy for c in self.components.values())
        self.power = np.clip(self.power + (total_energy / len(self.components) - 50) * 0.05, 0, 100)
        self.temperature += (self.power / 100) * 0.5 - (self.pressure / 30) * 0.2
        self.pressure += (self.power / 100) * 0.1 - (self.temperature / 1000) * 0.05
        self.history.append({
            'timestamp': datetime.now(),
            'power': self.power,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'core_phase': self.core_phase
        })

    def get_component_risks(self):
        return {name: dict(zip(["probability", "type"], comp.predict_risk()))
                for name, comp in self.components.items()}

class TauReactorControlSystem:
    def __init__(self, reactor_core):
        self.reactor = reactor_core
        self.emergency_shutdown_activated = False
        self.risk_threshold = 0.3
        self.control_log = deque(maxlen=100)
        self.learned_responses = {}

    def perform_control_cycle(self, delta_time):
        if self.emergency_shutdown_activated:
            self.reactor.power = max(0, self.reactor.power - 5)
            self.reactor.temperature = max(300, self.reactor.temperature - 10)
            self.log_control("Emergency Shutdown: Powering down")
            return

        self.reactor.update_core_state(delta_time)
        critical_components = []
        for name, comp in self.reactor.components.items():
            comp.update_state(delta_time, self.reactor.core_phase)
            risk_prob, risk_type = comp.predict_risk()
            if risk_prob >= self.risk_threshold:
                critical_components.append((name, risk_prob, risk_type))
                self.log_control(f"WARNING: Risk {name} ({risk_prob:.2f} {risk_type})")

        if critical_components:
            self.activate_preventative_measures(critical_components)

        if random.random() < 0.01:
            anomaly = random.choice(["coolant_leak", "control_rod_jam", "neutron_flux_spike"])
            self.handle_anomaly(anomaly, {"severity": random.uniform(0.1, 0.5)})

    def activate_preventative_measures(self, critical_components):
        self.log_control("Preventative measures activated.")
        for name, prob, r_type in critical_components:
            comp = self.reactor.components[name]
            if r_type == "OVERHEAT_RISK":
                comp.adjust_resonance(-0.02)
                self.log_control(f"Reduced resonance: {name}")
            elif r_type == "CRITICALITY_LOSS":
                comp.adjust_resonance(0.01)
                self.log_control(f"Increased resonance: {name}")
            if prob > 0.6:
                self.initiate_emergency_shutdown("HIGH_RISK_THRESHOLD")
                return

    def initiate_emergency_shutdown(self, reason):
        self.emergency_shutdown_activated = True
        self.log_control(f"EMERGENCY SHUTDOWN: {reason}")
        for comp in self.reactor.components.values():
            if comp.type == 'control_rod':
                comp.energy = 0
                comp.phase = math.pi
            if comp.type == 'coolant':
                comp.energy = 100

    def log_control(self, msg):
        self.control_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def handle_anomaly(self, anomaly, props):
        if anomaly not in self.learned_responses:
            self.learned_responses[anomaly] = self.create_anomaly_response(anomaly, props)
            self.log_control(f"Autogenesis created for: {anomaly}")
        self.learned_responses[anomaly](props)

    def create_anomaly_response(self, anomaly, props):
        if anomaly == "coolant_leak":
            def logic(p):
                self.log_control("Coolant leak — increasing coolant flow.")
                for c in self.reactor.components.values():
                    if c.type == 'coolant':
                        c.adjust_resonance(p['severity'] * 0.05)
                        c.energy = min(100, c.energy + p['severity'] * 20)
                self.reactor.temperature -= p['severity'] * 5
            return logic
        elif anomaly == "control_rod_jam":
            def logic(p):
                self.log_control("Control rod jam — compensating.")
                for c in self.reactor.components.values():
                    if c.type == 'control_rod':
                        c.adjust_resonance(-p['severity'] * 0.03)
                if p['severity'] > 0.3:
                    self.initiate_emergency_shutdown("CONTROL_ROD_JAM")
            return logic
        elif anomaly == "neutron_flux_spike":
            def logic(p):
                self.log_control("Neutron flux spike — reducing power.")
                self.reactor.power = max(0, self.reactor.power - p['severity'] * 30)
                self.reactor.temperature -= p['severity'] * 10
            return logic
        else:
            def default_logic(p):
                self.log_control(f"Unknown anomaly '{anomaly}' — cautious mode.")
                self.reactor.power *= 0.9
            return default_logic

class ReactorSimulator:
    def __init__(self):
        self.reactor = TauReactorCore("τ-Core-Reactor", 0.0)
        self.control_system = TauReactorControlSystem(self.reactor)
        self.setup()

    def setup(self):
        self.reactor.add_component(TauReactorEntity("Fuel_Rod_1", "fuel_rod", 0.5))
        self.reactor.add_component(TauReactorEntity("Coolant_1", "coolant", 0.3))
        self.reactor.add_component(TauReactorEntity("Control_Rod_1", "control_rod", 0.1))
        self.reactor.add_component(TauReactorEntity("Neutron_Reflector", "neutron", 0.8))

    def run(self, duration=100, step=0.5):
        print("\nτ-REACTOR SIMULATION START — Powered by UDT & UDP")
        t = 0.0
        while t < duration:
            self.control_system.perform_control_cycle(step)
            self.reactor.update_core_state(step)
            if int(t * 10) % 50 == 0:
                data = self.reactor.history[-1]
                print(f"{t:.1f}s | Power: {data['power']:.2f}% | Temp: {data['temperature']:.1f}°C | Pressure: {data['pressure']:.2f} MPa")
            if self.control_system.emergency_shutdown_activated and self.reactor.power == 0:
                print("Full shutdown reached.")
                break
            t += step
        print("Simulation complete.")
        for log in self.control_system.control_log:
            print(log)

if __name__ == "__main__":
    ReactorSimulator().run(duration=150)
