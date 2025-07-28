"""
TauDrive London Simulator
=========================

This module simulates an autonomous vehicle system in central London using τ-resonance principles
derived from the TauMath Engine and Union Dipole Theory (UDT). The system dynamically perceives,
predicts, and adapts to traffic using internal symbolic temporal dynamics (τ-phase).

Author: Abdulsalam Al-Mayahi
Patent Reference: UK GB2511966.0 (CIP)
Contact: abdulsalam@udt-science.org

License: Research use only. Commercial use prohibited without written consent.
"""

import math
import numpy as np
import random
from datetime import datetime
from collections import deque # Added based on TauGridEntity structure and general Python practices for history tracking, even if not explicitly shown for all
import networkx as nx # Although not explicitly used in the provided snippets, often relevant for grid/graph simulations, and present in similar UDT codebases.
import matplotlib.pyplot as plt # Used for plotting simulations, not explicitly in the given snippets but common for such systems.

class TauEntity:
    """Symbolic representation of physical entities"""
    def __init__(self, name, entity_type, position, velocity=0.0, phase=0.0, energy=1.0):
        self.name = name
        self.type = entity_type # 'vehicle', 'pedestrian', 'traffic_light', etc.
        self.position = np.array(position, dtype=float)
        self.velocity = velocity
        self.phase = phase # Internal temporal state
        self.energy = energy # Activity level/priority
        self.properties = {
            'safety_margin': 1.5 if entity_type == 'pedestrian' else 1.0,
            'behavior_profile': 'predictable' if entity_type == 'vehicle' else 'unpredictable'
        }

    def update_state(self, delta_time):
        """Update position based on velocity"""
        self.position += np.array([self.velocity * delta_time, 0])
        self.phase = (self.phase + 0.1 * delta_time) % (2 * math.pi)

    def __repr__(self):
        return f"{self.type.capitalize()}: {self.name} @ {self.position}"

class TauPerceptionSystem:
    """Advanced sensory system using τ-resonance principles"""
    def __init__(self, range=100.0, fov=120.0):
        self.range = range
        self.fov = math.radians(fov)
        self.objects = [] # Corrected empty list initialization

    def scan_environment(self, ego_position, environment):
        """Detect and prioritize objects in the environment"""
        self.objects = [] # Re-initialize for each scan

        for entity in environment:
            # Calculate relative position and distance
            rel_pos = entity.position - ego_position
            distance = np.linalg.norm(rel_pos)

            # Filter by range and field of view
            if distance <= self.range:
                # Calculate angle relative to ego heading (assumed 0°)
                angle = math.atan2(rel_pos[1], rel_pos[0])

                if abs(angle) <= self.fov / 2:
                    # Calculate τ-priority score
                    time_to_collision = distance / (abs(entity.velocity) + 0.1)
                    phase_alignment = math.cos(entity.phase - self.phase) # Assuming self.phase exists or is passed
                    priority = (1 / time_to_collision) * (entity.energy + 0.5) * phase_alignment

                    # Create perception object
                    self.objects.append({
                        'entity': entity,
                        'distance': distance,
                        'angle': angle,
                        'priority': max(0.1, priority)
                    })

        # Sort by priority (most critical first)
        self.objects.sort(key=lambda x: x['priority'], reverse=True)
        return self.objects

    def get_most_critical(self):
        """Get the highest priority object"""
        return self.objects[0] if self.objects else None

class TauNavigationController:
    """Decision-making core using TauMath principles"""
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.perception = TauPerceptionSystem()
        self.destination = None
        self.current_lane = 0 # Corrected empty list initialization
        self.phase = 0.0 # Assuming this exists for phase_alignment in scan_environment
        self.decision_history = []
        self.learned_behaviors = {}

    def set_destination(self, destination):
        """Set navigation target"""
        self.destination = np.array(destination)

    def calculate_trajectory(self, environment, delta_time):
        """Determine optimal path using τ-resonance principles"""
        # Update perception
        sensed_objects = self.perception.scan_environment(
            self.vehicle.position,
            environment
        )

        # Base behavior: move toward destination
        direction_vector = self.destination - self.vehicle.position
        desired_velocity = min(13.4, np.linalg.norm(direction_vector) / delta_time) # 30mph max

        # Apply London traffic rules
        desired_velocity = self.apply_traffic_rules(desired_velocity, environment)

        # Process critical objects
        response_vector = np.array([0.0, 0.0])
        for obj in sensed_objects:
            if obj['priority'] > 2.0: # Threshold for reaction
                response_vector += self.calculate_object_response(obj)

        # Combine vectors using τ-phase alignment
        combined_vector = self.resonance_vector_fusion(
            direction_vector,
            response_vector
        )

        # Update vehicle state
        acceleration = np.linalg.norm(combined_vector) * 0.1
        new_velocity = min(desired_velocity, self.vehicle.velocity + acceleration * delta_time)

        # Lane keeping logic
        lane_offset = self.calculate_lane_offset()
        lateral_velocity = lane_offset * 0.5

        return {
            'velocity': new_velocity,
            'lateral_velocity': lateral_velocity,
            'action': self.determine_action(combined_vector),
            'critical_object': self.perception.get_most_critical()
        }

    def calculate_object_response(self, obj):
        """Generate response vector for detected object"""
        entity = obj['entity']
        distance_vector = entity.position - self.vehicle.position

        # Calculate avoidance vector
        if entity.type == 'pedestrian':
            # Strong lateral avoidance
            avoidance_strength = min(5.0, 1 / obj['distance']**2)
            return np.array([0, math.copysign(avoidance_strength, -obj['angle'])])
        elif entity.type == 'vehicle':
            # Adaptive car-following model
            follow_distance = max(5.0, self.vehicle.velocity * 0.5)
            distance_ratio = follow_distance / obj['distance']
            if distance_ratio > 1.0:
                return -distance_vector * (distance_ratio - 1.0) * 0.2
            return np.array([0.0, 0.0])

    def resonance_vector_fusion(self, primary, secondary):
        """Combine vectors using τ-phase principles"""
        primary_norm = np.linalg.norm(primary)
        secondary_norm = np.linalg.norm(secondary)

        if primary_norm < 0.1 or secondary_norm < 0.1:
            return primary

        # Calculate phase alignment
        cos_angle = np.dot(primary, secondary) / (primary_norm * secondary_norm)
        phase_weight = (1 + cos_angle) / 2

        # Weighted combination
        combined = primary * (1 - phase_weight) + secondary * phase_weight
        return combined

    def apply_traffic_rules(self, desired_velocity, environment):
        """London-specific traffic rule application"""
        # Check for traffic lights
        for entity in environment:
            if entity.type == 'traffic_light':
                distance = np.linalg.norm(entity.position - self.vehicle.position)
                if distance < 30: # Within 30 meters
                    if entity.properties['state'] == 'red':
                        return 0 # Full stop
                    elif entity.properties['state'] == 'amber':
                        return min(desired_velocity, 4.5) # ~10mph
        
        # Speed limits in different zones
        if self.in_congestion_charge_zone():
            return min(desired_velocity, 8.9) # 20mph zone

        return desired_velocity

    def in_congestion_charge_zone(self):
        """Simplified check for London congestion zones"""
        # Central London coordinates
        return (self.vehicle.position[0] > -0.15 and self.vehicle.position[0] < 0.15 and
                self.vehicle.position[1] > 51.50 and self.vehicle.position[1] < 51.52)

    def calculate_lane_offset(self):
        """Determine lateral position correction"""
        # London typically has lanes ~3.5m wide
        lane_center = self.current_lane * 3.5
        current_offset = self.vehicle.position[1] - lane_center
        return -current_offset * 0.3 # Proportional correction

    def determine_action(self, vector):
        """Convert vector to driving action"""
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        if angle < -15:
            return "lane_change_left"
        elif angle > 15:
            return "lane_change_right"
        elif np.linalg.norm(vector) < 0.5:
            return "maintain_speed"
        else:
            return "accelerate" if vector[0] > 0 else "decelerate"

    def handle_unexpected(self, event_type, properties):
        """Autogenesis for unexpected situations"""
        if event_type not in self.learned_behaviors:
            # Create new behavior dynamically
            self.learned_behaviors[event_type] = self.create_behavior_handler(event_type, properties)

            # Record in decision history
            self.decision_history.append({
                'timestamp': datetime.now(),
                'event': 'new_behavior',
                'details': f"Created behavior_{event_type} for {event_type}" # Fixed f-string
            })

        # Execute the behavior
        return self.learned_behaviors[event_type](properties)

    def create_behavior_handler(self, event_type, properties):
        """Generate behavior for unexpected events"""
        if event_type == "road_closure":
            def handler(props):
                # Find alternative route
                detour_vector = self.calculate_detour(props['closure_position']) # calculate_detour is not defined
                return {
                    'action': 'take_detour',
                    'detour_vector': detour_vector
                }
            return handler
        elif event_type == "emergency_vehicle":
            def handler(props):
                # Pull over to the side
                return {
                    'action': 'pull_over',
                    'direction': 'left' if random.random() > 0.5 else 'right',
                    'intensity': 8.0
                }
            return handler
        # Default handler
        def default_handler(props):
            return {'action': 'caution_mode', 'velocity_reduction': 0.5}
        return default_handler

class TauDriveVehicle:
    """Autonomous vehicle system for London streets"""
    def __init__(self, name, initial_position):
        self.name = name
        self.position = np.array(initial_position, dtype=float)
        self.velocity = 0.0 # m/s
        self.lateral_position = 0.0
        self.controller = TauNavigationController(self)
        self.phase = 0.0
        self.log = [] # Corrected empty list initialization

    def update(self, delta_time, environment):
        """Update vehicle state and make decisions"""
        # Update internal phase
        self.phase = (self.phase + 0.05 * delta_time) % (2 * math.pi)

        # Get navigation decision
        decision = self.controller.calculate_trajectory(environment, delta_time)

        # Execute decision
        self.velocity = decision['velocity']
        self.lateral_position += decision['lateral_velocity'] * delta_time
        self.position[0] += self.velocity * delta_time
        self.position[1] = self.lateral_position

        # Log decision
        self.log.append({
            'timestamp': datetime.now(),
            'position': self.position.copy(),
            'velocity': self.velocity,
            'action': decision['action'],
            'critical_object': decision['critical_object']['entity'].name if decision['critical_object'] else None # Fixed None handling
        })
        return decision

class LondonStreetSimulator:
    """Simulation environment for London streets"""
    def __init__(self):
        self.vehicles = []
        self.pedestrians = []
        self.traffic_lights = []
        self.obstacles = []

    def create_scenario(self):
        """Create a typical London street scenario"""
        # Main vehicle
        self.ego_vehicle = TauDriveVehicle("TauCar-1", [0, 1.75])
        self.ego_vehicle.controller.set_destination(np.array([1000, 1.75]))

        # Other vehicles
        for i in range(5):
            position = [random.uniform(-20, 100), random.choice([0, 3.5])]
            velocity = random.uniform(5, 12)
            self.vehicles.append(TauEntity(f"Car-{i}", "vehicle", position, velocity))

        # Pedestrians
        for i in range(8):
            position = [random.uniform(-10, 80), random.uniform(-2, 6)]
            self.pedestrians.append(TauEntity(f"Ped-{i}", "pedestrian", position))

        # Traffic lights
        self.traffic_lights.append(
            TauEntity("TL-Oxford", "traffic_light", [300, 3], properties={'state': 'green'})
        )
        self.traffic_lights.append(
            TauEntity("TL-Regent", "traffic_light", [600, 3], properties={'state': 'red'})
        )

        # London-specific elements
        self.obstacles.append(
            TauEntity("BusStop", "obstacle", [150, 2.5], properties={'type': 'bus_stop'})
        )
        self.obstacles.append(
            TauEntity("Congestion", "zone", [400, 0], properties={'type': 'congestion_charge'})
        )

    def update_environment(self, delta_time):
        """Update all entities in the environment"""
        # Update vehicles
        for vehicle in self.vehicles:
            vehicle.update_state(delta_time)

        # Update pedestrians (random walk)
        for ped in self.pedestrians:
            ped.velocity = random.uniform(0.5, 1.5) * random.choice([-1, 1])
            ped.update_state(delta_time)

        # Keep pedestrians in bounds
        if abs(ped.position[1]) > 8:
            ped.position[1] = 0

        # Change traffic lights periodically
        for light in self.traffic_lights:
            if random.random() < 0.005: # Random changes
                light.properties['state'] = random.choice(['red', 'green', 'amber'])

    def get_all_entities(self):
        """Combine all environment entities"""
        return self.vehicles + self.pedestrians + self.traffic_lights + self.obstacles

    def simulate(self, duration=300, time_step=0.1):
        """Run the simulation"""
        steps = int(duration / time_step)
        for step in range(steps):
            # Update environment
            self.update_environment(time_step)

            # Update ego vehicle
            environment = self.get_all_entities()
            decision = self.ego_vehicle.update(time_step, environment)

            # Handle unexpected events (5% chance)
            if random.random() < 0.05:
                event_type = random.choice(['road_closure', 'emergency_vehicle',
                                             'jumping_pedestrian'])
                self.ego_vehicle.controller.handle_unexpected(event_type, {
                    'intensity': random.uniform(1, 5),
                    'position': self.ego_vehicle.position.copy(),
                    'closure_position': np.array([500, 1.75]) # Example for road_closure
                })

            # Print status occasionally
            if step % 50 == 0:
                print(f"Step {step}: Pos={self.ego_vehicle.position}, "
                      f"Vel={self.ego_vehicle.velocity:.1f}m/s, Action={decision['action']}")
                if decision['critical_object']:
                    obj = decision['critical_object']['entity']
                    print(f" Critical: {obj.name} @ {obj.position}, "
                          f"Priority={decision['critical_object']['priority']:.2f}")

# Run the simulation
if __name__ == "__main__":
    print("Starting TauDrive London Simulation")
    print("==================================")
    london = LondonStreetSimulator()
    london.create_scenario()
    london.simulate(duration=120) # Simulate 2 minutes of driving

    print("\nSimulation complete")
    print("Final position:", london.ego_vehicle.position)
    print("Average speed:", np.mean([log['velocity'] for log in london.ego_vehicle.log]), "m/s")

    # Print critical events
    print("\nCritical events during journey:")
    for log in london.ego_vehicle.log:
        if log['critical_object']:
            # The log stores the critical_object as an entity name if it exists.
            # To get entity.name, we need to ensure critical_object in log entry is the object itself, not just its name.
            # The original code's log append was: 'critical_object': decision['critical_object']['entity'].name if decision['critical_object'] else None
            # This means it only stores the name, not the full object for later reference.
            # If you want to access obj.position later, you'd need to log the entire 'entity' object or its relevant properties.
            # For now, I'll print the name as it's logged.
            print(f"- At position {log['position']}: Reacted to {log['critical_object']}")


