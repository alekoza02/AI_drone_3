from neural_network import *
from drone import *

import pickle
from random import random, randint
from math import sin, cos, radians, sqrt
from time import perf_counter_ns
import copy


class Arena:
    def __init__(self, n_agents=10, n_epoch=10, max_steps=200, drone_starting_position=[0, 0]):
        
        self.n_agents = n_agents
        self.n_epoch = n_epoch
        self.max_steps = max_steps
        self.drone_starting_position = drone_starting_position
        self.current_epoch = 0
        self.training_finished = False
        self.force_new_epoch = False
        self.best_prev_score = 0

        self.arena_size = [400, 300]
        self.destinations = None
        
        self.start_next_epoch()


    def set_goal(self, goal):
        self.destinations = goal
        [drone.set_destination(self.destinations[0]) for drone in self.agents_drone]


    def set_simulation_world_size(self, size):
        self.arena_size = size

    
    def train_single_step(self):

        self.current_step += 1 
        
        if self.current_step >= self.max_steps or sum(self.agents_mask) == 0:
            self.force_new_epoch = True
            
            self.best_scores_indices = sorted(
                range(self.n_agents),
                key=lambda i: self.agents_scores[i],
                reverse=True
            )

            self.best_prev_score = self.agents_scores[self.best_scores_indices[0]]

            return
        
        for agent, drone, index in zip(self.agents_NN, self.agents_drone, range(self.n_agents)):
            if self.agents_mask[index] and (drone.pos[0] < - self.arena_size[0] * 0.2 or drone.pos[0] > self.arena_size[0] * 1.2 or drone.pos[1] < - self.arena_size[1] * 0.2 or drone.pos[1] > self.arena_size[1] * 1.2):
                self.agents_scores[index] += -0.1
                self.agents_mask[index] = False
                
            if self.agents_mask[index]:

                if self.current_step == (self.max_steps - 1):
                    self.agents_scores[index] += 10 # alive bonus

                delta_ia, delta_physics = self.raw_step(drone, agent)

                self.timings_ia += delta_ia
                self.timings_physics += delta_physics
                self.timings_sample += 1

                self.agents_scores[index] += self.compute_reward(drone)


    def compute_reward(self, drone: Drone):
        # Parameters
        target_radius = 10.0          # radius around the target to consider "at target"
        target_time = 5.0             # time to accumulate full reward at target

        # Distance to current target
        dx = drone.destination[0] - drone.pos[0]
        dy = drone.destination[1] - drone.pos[1]
        dist = (dx**2 + dy**2)**0.5

        # Base reward: inverse distance
        r = 1.0 / (1.0 + dist)

        # # Incremental "time at target" tracking
        if dist < target_radius:
            drone.steps_at_destination += 1
            # If enough time spent at target, advance to next
            if drone.steps_at_destination >= target_time:
                drone.destinations_reached += 1
                if drone.destinations_reached < len(self.destinations):
                    r += 10
                    drone.set_destination(self.destinations[drone.destinations_reached])
                drone.steps_at_destination = 0
        else:
            # Not at target, reset accumulated time
            drone.steps_at_destination = 0

        return r * math.cos(math.radians(drone.rotation)) ** 2



    def start_next_epoch(self):
            
        # self.set_goal([random() * self.arena_size[0], random() * self.arena_size[1]])

        self.evolve()

        self.agents_drone = [Drone(self.drone_starting_position[0], self.drone_starting_position[1]) for _ in range(self.n_agents)]
        if not self.destinations is None:
            [drone.set_destination(self.destinations[0]) for drone in self.agents_drone]
        self.agents_scores = [0 for _ in range(self.n_agents)]
        self.best_scores_indices = [0 for i in range(self.n_agents)]
        self.agents_mask = [True for _ in range(self.n_agents)]
        
        self.timings_ia = 0
        self.timings_physics = 0
        self.timings_sample = 0.00001

        self.force_new_epoch = False
        self.current_step = 0

        self.max_inputs = [-1e6 for _ in range(7)]
        self.min_inputs = [1e6 for _ in range(7)]
        self.max_outputs = [-1e6 for _ in range(7)]
        self.min_outputs = [1e6 for _ in range(7)]


    def evolve(self):

        if self.current_epoch == 0:
            self.agents_NN = [NeuralNetwork() for _ in range(self.n_agents)]
            return

        N = self.n_agents
        ELITES = max(4, int(self.n_agents * 0.1))
        TOURNAMENT_K = 5

        MUT_P = 0.02          # per-parameter mutation probability
        MUT_SIGMA = 0.005      # mutation std
        W_MAX = 3.0           # weight/bias clamp

        # ---------- helpers ----------

        def tournament_select(ranked):
            best_i = None
            best_score = -1e18
            for _ in range(TOURNAMENT_K):
                i = randint(0, N - 1)
                if self.agents_scores[ranked[i]] > best_score:
                    best_score = self.agents_scores[ranked[i]]
                    best_i = i
            return self.agents_NN[ranked[best_i]]

        def crossover(a, b):
            child = copy.deepcopy(a)

            for wA, wB, wC in zip(a.input_layer_w, b.input_layer_w, child.input_layer_w):
                for i in range(len(wC)):
                    wC[i] = wA[i] if random() < 0.5 else wB[i]

            for wA, wB, wC in zip(a.hidden_layer1_w, b.hidden_layer1_w, child.hidden_layer1_w):
                for i in range(len(wC)):
                    wC[i] = wA[i] if random() < 0.5 else wB[i]

            for wA, wB, wC in zip(a.hidden_layer2_w, b.hidden_layer2_w, child.hidden_layer2_w):
                for i in range(len(wC)):
                    wC[i] = wA[i] if random() < 0.5 else wB[i]

            for i in range(len(child.hidden_layer1_b)):
                child.hidden_layer1_b[i] = (
                    a.hidden_layer1_b[i] if random() < 0.5 else b.hidden_layer1_b[i]
                )

            for i in range(len(child.hidden_layer2_b)):
                child.hidden_layer2_b[i] = (
                    a.hidden_layer2_b[i] if random() < 0.5 else b.hidden_layer2_b[i]
                )

            for i in range(len(child.output_layer_b)):
                child.output_layer_b[i] = (
                    a.output_layer_b[i] if random() < 0.5 else b.output_layer_b[i]
                )

            return child

        def mutate(nn):
            def mutate_matrix(M):
                for i in range(len(M)):
                    for j in range(len(M[i])):
                        if random() < MUT_P:
                            M[i][j] += (random() * 2 - 1) * MUT_SIGMA
                            M[i][j] = max(-W_MAX, min(W_MAX, M[i][j]))

            def mutate_vector(v):
                for i in range(len(v)):
                    if random() < MUT_P:
                        v[i] += (random() * 2 - 1) * MUT_SIGMA
                        v[i] = max(-W_MAX, min(W_MAX, v[i]))

            mutate_matrix(nn.input_layer_w)
            mutate_matrix(nn.hidden_layer1_w)
            mutate_matrix(nn.hidden_layer2_w)

            mutate_vector(nn.hidden_layer1_b)
            mutate_vector(nn.hidden_layer2_b)
            mutate_vector(nn.output_layer_b)

        # ---------- evolution ----------

        ranked = sorted(
            range(N),
            key=lambda i: self.agents_scores[i],
            reverse=True
        )

        new_population = []

        # elites
        for i in range(ELITES):
            new_population.append(copy.deepcopy(self.agents_NN[ranked[i]]))

        # offspring
        while len(new_population) < N:
            p1 = tournament_select(ranked)
            p2 = tournament_select(ranked)
            child = crossover(p1, p2)
            mutate(child)
            new_population.append(child)

        self.agents_NN = new_population



    def raw_step(self, drone, agent, step_dt=1):

        inputs = [
                (drone.destination[0] - drone.pos[0]) / self.arena_size[0],   # 1) to target X        
                (drone.destination[1] - drone.pos[1]) / self.arena_size[1],   # 2) to target Y        
                drone.speed[0] / 200,                          # 3) speed X        
                drone.speed[1] / 200,                          # 4) speed Y        
                cos(radians(drone.rotation)),                   # 5) cos angle        
                sin(radians(drone.rotation)),                   # 6) sin angle
                drone.angular_velocity / 2000,                         # 7) angular speed
            ]

        # DEBUGGING
        self.max_inputs = [
            max(i, j) for i, j in zip(self.max_inputs, inputs)
        ]
        self.min_inputs = [
            min(i, j) for i, j in zip(self.min_inputs, inputs)
        ]

        start_ia = perf_counter_ns()
        output = agent.IA_step(inputs)
        stop_ia = perf_counter_ns()
        
        # DEBUGGING
        self.max_outputs = [
            max(i, j) for i, j in zip(self.max_outputs, output)
        ]
        self.min_outputs = [
            min(i, j) for i, j in zip(self.min_outputs, output)
        ]
        
        # TODO ---> Need appropriate mapping
        drone.thrusters_power = [drone.thrusters_power[0] + min(0.2, max(-0.2, (output[0] + 1) * 0.5 - drone.thrusters_power[0])), drone.thrusters_power[1] + min(0.2, max(-0.2, (output[2] + 1) * 0.5 - drone.thrusters_power[1]))]
        drone.thrustets_rotations_local = [drone.thrustets_rotations_local[0] + min(15, max(-15, output[1] * 45 - drone.thrustets_rotations_local[0])), drone.thrustets_rotations_local[1] + min(15, max(-15, output[3] * 45 - drone.thrustets_rotations_local[1]))]

        start_physics = perf_counter_ns()    
        _ = drone.physics_simulation_step(step_dt)
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
        

    def populate_with_loaded_NN(self, agent):
        self.agents_NN = [copy.deepcopy(agent) for _ in range(self.n_agents)]