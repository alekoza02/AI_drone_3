import math

class Drone:
    def __init__(self, x, y):

        self.reached_in_steps = 1e6

        self.size = 50
        self.mass = self.size * 2
        self.thrusters_multiplier = 100
        self.gravity = .981
        self.pos = [x, y]
        self.speed = [0, 0]
        self.rotation = 0

        self.thrustets_center = [
            [self.pos[0] - self.size_perc(0.7), self.pos[1]],
            [self.pos[0] + self.size_perc(0.7), self.pos[1]]
        ]
        self.thrusters_power = [0, 0]
        self.thrustets_rotations_local = [0, 0]
        self.thrustets_rotations_global = [0, 0]
        
        self.angular_velocity = 0

        self.color_property_value = 1

        self.debug_visuals = []

    
    def size_perc(self, x):
        return self.size * x


    def get_drone_visual_info(self):
        return {
            'corp' : {
                'verteces' : [
                    [self.pos[0] - self.size_perc(0.5), self.pos[1] - self.size_perc(0.5)], 
                    [self.pos[0] - self.size_perc(0.5), self.pos[1] + self.size_perc(0.5)], 
                    [self.pos[0] + self.size_perc(0.5), self.pos[1] + self.size_perc(0.5)], 
                    [self.pos[0] + self.size_perc(0.5), self.pos[1] - self.size_perc(0.5)], 
                ],
                'color' : [100, 100, 100],
                'global_rotation' : self.rotation,
                'local_rotation' : 0,
                'global_position' : self.pos,
                'local_position' : self.pos, 
            },
            'thr_1' : {
                'verteces' : [
                    [self.pos[0] - self.size_perc(0.5) - self.size_perc(0.1) - self.size_perc(0.2), self.pos[1] - self.size_perc(0.3)],
                    [self.pos[0] - self.size_perc(0.5) - self.size_perc(0.1) - self.size_perc(0.2), self.pos[1] + self.size_perc(0.3)],
                    [self.pos[0] - self.size_perc(0.5) - self.size_perc(0.1), self.pos[1] + self.size_perc(0.3)],
                    [self.pos[0] - self.size_perc(0.5) - self.size_perc(0.1), self.pos[1] - self.size_perc(0.3)],
                ],
                'color' : [1 if self.thrusters_power[0] > 1 else 0, self.thrusters_power[0] if self.thrusters_power[0] < 1 else 0, 0],
                'global_rotation' : self.rotation,
                'local_rotation' : self.thrustets_rotations_local[0],
                'global_position' : self.pos,
                'local_position' : self.thrustets_center[0], 
            },
            'thr_2' : {
                'verteces' : [
                    [self.pos[0] + self.size_perc(0.5) + self.size_perc(0.1) + self.size_perc(0.2), self.pos[1] - self.size_perc(0.3)],
                    [self.pos[0] + self.size_perc(0.5) + self.size_perc(0.1) + self.size_perc(0.2), self.pos[1] + self.size_perc(0.3)],
                    [self.pos[0] + self.size_perc(0.5) + self.size_perc(0.1), self.pos[1] + self.size_perc(0.3)],
                    [self.pos[0] + self.size_perc(0.5) + self.size_perc(0.1), self.pos[1] - self.size_perc(0.3)],
                ],
                'color' : [1 if self.thrusters_power[1] > 1 else 0, self.thrusters_power[1] if self.thrusters_power[1] < 1 else 0, 0],
                'global_rotation' : self.rotation,
                'local_rotation' : self.thrustets_rotations_local[1],
                'global_position' : self.pos,
                'local_position' : self.thrustets_center[1], 
            }
        }
    

    def update_global_thruster_rotation(self):
        self.thrustets_rotations_global[0] = self.rotation + self.thrustets_rotations_local[0]
        self.thrustets_rotations_global[1] = self.rotation + self.thrustets_rotations_local[1]


    def physics_simulation_step(self):

        self.update_global_thruster_rotation()

        # TRASLATION        
        thr_1_x = - self.thrusters_power[0] * math.cos(math.radians(self.thrustets_rotations_global[0]) + math.pi / 2)
        thr_1_y = - self.thrusters_power[0] * math.sin(math.radians(self.thrustets_rotations_global[0]) + math.pi / 2)
        thr_2_x = - self.thrusters_power[1] * math.cos(math.radians(self.thrustets_rotations_global[1]) + math.pi / 2)
        thr_2_y = - self.thrusters_power[1] * math.sin(math.radians(self.thrustets_rotations_global[1]) + math.pi / 2)

        self.speed[0] += self.thrusters_multiplier * (thr_1_x + thr_2_x) / self.mass
        self.speed[1] += self.thrusters_multiplier * (thr_1_y + thr_2_y) / self.mass + self.gravity

        delta_space_vector = [
            self.speed[0],
            self.speed[1],
        ]

        self.pos[0] += delta_space_vector[0]
        self.pos[1] += delta_space_vector[1]

        self.thrustets_center[0][0] += delta_space_vector[0]
        self.thrustets_center[0][1] += delta_space_vector[1]
        self.thrustets_center[1][0] += delta_space_vector[0]
        self.thrustets_center[1][1] += delta_space_vector[1]


        # ROTATION
        thr_1_tg = - self.thrusters_power[0] * math.sin(math.radians(self.thrustets_rotations_local[0]) + math.pi / 2)
        thr_2_tg = - self.thrusters_power[1] * math.sin(math.radians(self.thrustets_rotations_local[1]) + math.pi / 2)

        torque = (thr_2_tg - thr_1_tg)

        self.rotation += torque

        self.angular_velocity += torque 

        torque_cos = math.cos(math.radians(torque))
        torque_sin = math.sin(math.radians(torque))

        x1 = self.thrustets_center[0][0] - self.pos[0]
        y1 = self.thrustets_center[0][1] - self.pos[1]  
        x2 = self.thrustets_center[1][0] - self.pos[0]
        y2 = self.thrustets_center[1][1] - self.pos[1]

        self.thrustets_center[0][0] = + x1 * torque_cos - y1 * torque_sin + self.pos[0]
        self.thrustets_center[0][1] = + x1 * torque_sin + y1 * torque_cos + self.pos[1]
        self.thrustets_center[1][0] = + x2 * torque_cos - y2 * torque_sin + self.pos[0]
        self.thrustets_center[1][1] = + x2 * torque_sin + y2 * torque_cos + self.pos[1]
                
        self.thrustets_rotations_global[0] += torque
        self.thrustets_rotations_global[1] += torque
            
        self.debug_visuals = [delta_space_vector, [thr_1_x, thr_1_y], [thr_2_x, thr_2_y]]

