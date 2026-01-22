from neural_network import *
from drone import *

from math import sin, cos, radians
from time import perf_counter_ns

class Arena:
    def __init__(self, n_agents=10, n_epoch=10, max_steps=200, drone_starting_position=[0, 0]):
        self.n_agents = n_agents
        self.n_epoch = n_epoch

        self.agents_NN = [NeuralNetwork() for _ in range(self.n_agents)]
        self.agents_drone = [Drone(drone_starting_position[0], drone_starting_position[1]) for _ in range(self.n_agents)]
        self.agents_scores = [[] for _ in range(self.n_agents)]
        self.best_scores_indices = []
        self.agents_mask = [True for _ in range(self.n_agents)]

        self.timings_ia = 0
        self.timings_physics = 0
        self.timings_sample = 0.00001

        self.destination = [0, 0]
        self.arena_size = [400, 300]

        self.max_steps = max_steps
        self.current_step = 0


    def set_goal(self, goal):
        self.destination = goal


    def set_simulation_world_size(self, size):
        self.arena_size = size

    
    def train_single_step(self):

        if self.current_step > self.max_steps:
            return
        
        self.current_step += 1 

        for agent, drone, index in zip(self.agents_NN, self.agents_drone, range(self.n_agents)):
            if drone.pos[0] < 0 or drone.pos[0] > self.arena_size[0] or drone.pos[1] < 0 or drone.pos[1] > self.arena_size[1]:
                self.agents_mask[index] = False
                
            if self.agents_mask[index]:

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
                
                drone.thrusters_power = [output[0], output[2]]
                drone.thrustets_rotations_local = [output[1] * 10, output[3] * 10]

                start_physics = perf_counter_ns()    
                _ = drone.physics_simulation_step()
                stop_physics = perf_counter_ns()

                self.timings_ia += stop_ia - start_ia
                self.timings_physics += stop_physics - start_physics
                self.timings_sample += 1

                
                distance = ((self.destination[0] - drone.pos[0]) ** 2 + (self.destination[1] - drone.pos[1]) ** 2) ** 0.5
                stay_at_point = distance < 10 # pixel units
                if stay_at_point:
                    drone.reached_in_steps = min(self.current_step, drone.reached_in_steps)

                self.agents_scores[index].append(1 / distance + 1 * stay_at_point + 10 / drone.reached_in_steps)

                self.best_scores_indices = sorted(range(len(self.agents_scores)), key=[sum(i) for i in self.agents_scores].__getitem__, reverse=True)


    def start_next_epoch(self):
        ...