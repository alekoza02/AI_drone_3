from neural_network import *
from drone import *

import pickle
from random import random
from math import sin, cos, radians
from time import perf_counter_ns

class Arena:
    def __init__(self, n_agents=10, n_epoch=10, max_steps=200, n_mutations_per_evolution=1, drone_starting_position=[0, 0]):
        
        self.n_mutations_per_evolution = n_mutations_per_evolution
        self.n_agents = n_agents
        self.n_epoch = n_epoch
        self.max_steps = max_steps
        self.drone_starting_position = drone_starting_position
        self.current_epoch = 0
        self.training_finished = False
        self.force_new_epoch = False

        self.arena_size = [400, 300]
        self.destination = [0, 0]
        
        self.start_next_epoch()



    def set_goal(self, goal):
        self.destination = goal


    def set_simulation_world_size(self, size):
        self.arena_size = size

    
    def train_single_step(self):

        self.current_step += 1 
        
        if self.current_step >= self.max_steps or sum(self.agents_mask) == 0:
            self.force_new_epoch = True
            return
        
        for agent, drone, index in zip(self.agents_NN, self.agents_drone, range(self.n_agents)):
            if drone.pos[0] < - self.arena_size[0] * 0.2 or drone.pos[0] > self.arena_size[0] * 1.2 or drone.pos[1] < - self.arena_size[1] * 0.2 or drone.pos[1] > self.arena_size[1] * 1.2:
                self.agents_mask[index] = False
                
            if self.agents_mask[index]:

                delta_ia, delta_physics = self.raw_step(drone, agent)

                self.timings_ia += delta_ia
                self.timings_physics += delta_physics
                self.timings_sample += 1

                                
                distance = ((self.destination[0] - drone.pos[0])**2 + (self.destination[1] - drone.pos[1])**2)**0.5

                if distance < 2:
                    fitness = (self.arena_size[0] ** 2 + self.arena_size[1] ** 2) ** 0.5 - self.current_step          # fast arrival dominates
                elif not self.agents_mask[index]:
                    fitness = -(self.arena_size[0] ** 2 + self.arena_size[1] ** 2) ** 0.5                              # crash penalty
                else:
                    fitness = -distance                         # dense shaping toward target

                fitness *= cos(radians(drone.rotation)) ** 2

                self.agents_scores[index] = fitness

                self.best_scores_indices = sorted(
                    range(self.n_agents),
                    key=lambda i: self.agents_scores[i],
                    reverse=True
                )

    def start_next_epoch(self):
            
        self.set_goal([random() * self.arena_size[0], random() * self.arena_size[1]])

        self.agents_drone = [Drone(self.drone_starting_position[0], self.drone_starting_position[1]) for _ in range(self.n_agents)]
        self.agents_scores = [0 for _ in range(self.n_agents)]
        self.best_scores_indices = [0 for i in range(self.n_agents)]
        self.agents_mask = [True for _ in range(self.n_agents)]

        self.evolve()

        
        self.timings_ia = 0
        self.timings_physics = 0
        self.timings_sample = 0.00001

        self.force_new_epoch = False
        self.current_step = 0


    def evolve(self):

        if self.current_epoch == 0:
            self.agents_NN = [NeuralNetwork() for _ in range(self.n_agents)]

        else:

            import copy; order = [copy.deepcopy(self.agents_NN[i]) for i in self.best_scores_indices]

            for i in range(self.n_mutations_per_evolution):
                for i in range(int(0.3 * self.n_agents), self.n_agents):
                    
                    layer_choice = random()
                    node_choice = random()
                    link_choice = random()

                    if layer_choice < 0.33:
                        
                        # bias
                        layer_size = 9
                        node_index = int(node_choice * layer_size)
                        order[i].hidden_layer1_b[node_index] = max(-2.0, min(2.0, order[i].hidden_layer1_b[node_index] + (random() - 0.5) * 0.1))

                        # weight
                        layer_size = 4
                        link_size = 9
                        node_index = int(node_choice * layer_size)
                        link_index = int(link_choice * link_size)
                        order[i].input_layer_w[node_index][link_index] += max(-2.0, min(2.0, order[i].input_layer_w[node_index][link_index] + (random() - 0.5) * 0.1))
                                
                    elif layer_choice < 0.66:
                        
                        # bias
                        layer_size = 9        
                        node_index = int(node_choice * layer_size)
                        order[i].hidden_layer2_b[node_index] = max(-2.0, min(2.0, order[i].hidden_layer2_b[node_index] + (random() - 0.5) * 0.1))

                        # weight
                        layer_size = 9
                        link_size = 9
                        node_index = int(node_choice * layer_size)
                        link_index = int(link_choice * link_size)
                        order[i].hidden_layer1_w[node_index][link_index] += max(-2.0, min(2.0, order[i].hidden_layer1_w[node_index][link_index] + (random() - 0.5) * 0.1))
                    
                    elif layer_choice < 1:
                        
                        # bias
                        layer_size = 4        
                        node_index = int(node_choice * layer_size)
                        order[i].output_layer_b[node_index] = max(-2.0, min(2.0, order[i].output_layer_b[node_index] + (random() - 0.5) * 0.1))

                        # weight
                        layer_size = 9
                        link_size = 4
                        node_index = int(node_choice * layer_size)
                        link_index = int(link_choice * link_size)
                        order[i].hidden_layer2_w[node_index][link_index] += max(-2.0, min(2.0, order[i].hidden_layer2_w[node_index][link_index] + (random() - 0.5) * 0.1))

            self.agents_NN = order


    def raw_step(self, drone, agent):
        inputs = [
            (self.destination[0] - drone.pos[0]) / self.arena_size[0],   # 1) to target X        
            (self.destination[1] - drone.pos[1]) / self.arena_size[1],   # 2) to target Y        
            drone.speed[0] / 20,                          # 3) speed X        
            drone.speed[1] / 20,                          # 4) speed Y        
            cos(radians(drone.rotation)),                   # 5) cos angle        
            sin(radians(drone.rotation)),                   # 6) sin angle
            drone.angular_velocity,                         # 7) angular speed
        ]


        start_ia = perf_counter_ns()
        output = agent.IA_step(inputs)
        stop_ia = perf_counter_ns()
        
        # TODO ---> Need appropriate mapping
        drone.thrusters_power = [min(1, (max(0, output[0] * 0.05 + drone.thrusters_power[0]))), min(1, (max(0, output[2] * 0.05 + drone.thrusters_power[1])))]
        drone.thrustets_rotations_local = [output[1] * 0.01 + drone.thrustets_rotations_local[0], output[3] * 0.01 + drone.thrustets_rotations_local[1]]

        start_physics = perf_counter_ns()    
        _ = drone.physics_simulation_step()
        stop_physics = perf_counter_ns()

        return stop_ia - start_ia, stop_physics - start_physics    


    def save_best_NN(self, filename):
        """Save a class instance to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self.agents_NN[self.best_scores_indices[0]], f)


    @staticmethod
    def load_best_NN(filename):
        """Load a class instance from a file using pickle."""
        with open(filename, 'rb') as f:
            return pickle.load(f)