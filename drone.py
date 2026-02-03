import math
import random

gravity = 9.81

class Drone:
    def __init__(self, x, y, no_smoke=True):

        self.no_smoke = no_smoke
        self.reached_in_steps = 1e6

        self.destination = [x, y]
        self.steps_at_destination = 0
        self.destinations_reached = 0
        self.prev_step_validation = 0
        self.now_step_validation = 0

        self.size = 50
        self.mass = self.size * 2
        self.thrusters_multiplier = 1000
        self.pos = [x, y]
        self.speed = [0, 0]
        self.rotation = 0

        self.radius_application_point = self.size_perc(0.7)
        self.thrustets_center = [
            [self.pos[0] - self.radius_application_point, self.pos[1]],
            [self.pos[0] + self.radius_application_point, self.pos[1]]
        ]
        self.thrusters_power = [0, 0]
        self.thrustets_rotations_local = [0, 0]
        self.thrustets_rotations_global = [0, 0]
        
        self.angular_velocity = 0

        self.color_property_value = 1

        self.debug_visuals = []

        self.reactor_particles = []

    
    def set_destination(self, destination):
        self.destination = destination
        self.steps_at_destination = 0


    def update_step_validation(self, current_step):

        if abs(self.prev_step_validation - current_step) > 1:
            self.steps_at_destination = 0

        self.prev_step_validation = self.now_step_validation
        self.now_step_validation = current_step


    def size_perc(self, x):
        return self.size * x


    def get_drone_visual_info(self):
        return {
            'particles' : {
                'coords' : [[i.x, i.y] for i in self.reactor_particles],
                'life_step' : [i.life_step for i in self.reactor_particles],
                'max_life_step' : [i.life_max_step for i in self.reactor_particles],
                'radius' : [i.radius for i in self.reactor_particles],
                'ini_radius' : [i.radius0 for i in self.reactor_particles],
                'orientation' : [i.orientation for i in self.reactor_particles]
            },
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


    def physics_simulation_step(self, step_dt):
        # --- UPDATE THRUSTER GLOBAL ROTATION ---
        self.update_global_thruster_rotation()

        # =========================
        # TRANSLATION (LINEAR)
        # =========================

        # Thrust force components (world frame)
        thr_1_x = -self.thrusters_power[0] * math.cos(
            math.radians(self.thrustets_rotations_global[0]) + math.pi / 2
        )
        thr_1_y = -self.thrusters_power[0] * math.sin(
            math.radians(self.thrustets_rotations_global[0]) + math.pi / 2
        )

        thr_2_x = -self.thrusters_power[1] * math.cos(
            math.radians(self.thrustets_rotations_global[1]) + math.pi / 2
        )
        thr_2_y = -self.thrusters_power[1] * math.sin(
            math.radians(self.thrustets_rotations_global[1]) + math.pi / 2
        )

        ax = self.thrusters_multiplier * (thr_1_x + thr_2_x) / self.mass
        ay = self.thrusters_multiplier * (thr_1_y + thr_2_y) / self.mass + gravity

        self.speed[0] += ax * step_dt
        self.speed[1] += ay * step_dt

        dx = self.speed[0] * step_dt
        dy = self.speed[1] * step_dt

        self.pos[0] += dx
        self.pos[1] += dy

        # Move thruster centers with body
        for i in range(2):
            self.thrustets_center[i][0] += dx
            self.thrustets_center[i][1] += dy

        # =========================
        # ROTATION (ANGULAR)
        # =========================

        thr_1_tg = -self.thrusters_power[0] * math.sin(
            math.radians(self.thrustets_rotations_local[0]) + math.pi / 2
        )
        thr_2_tg = -self.thrusters_power[1] * math.sin(
            math.radians(self.thrustets_rotations_local[1]) + math.pi / 2
        )

        torque = (thr_2_tg - thr_1_tg) * self.radius_application_point * 10

        moment_of_inertia = self.mass * (self.size ** 2) / 12

        angular_acceleration = torque / moment_of_inertia

        self.angular_velocity += angular_acceleration * step_dt

        rotation_delta = self.angular_velocity * step_dt
        self.rotation += rotation_delta

        x1 = self.thrustets_center[0][0] - self.pos[0]  
        y1 = self.thrustets_center[0][1] - self.pos[1]  
        x2 = self.thrustets_center[1][0] - self.pos[0]
        y2 = self.thrustets_center[1][1] - self.pos[1]

        torque_cos = math.cos(math.radians(rotation_delta))
        torque_sin = math.sin(math.radians(rotation_delta))

        self.thrustets_center[0][0] = + x1 * torque_cos - y1 * torque_sin + self.pos[0]
        self.thrustets_center[0][1] = + x1 * torque_sin + y1 * torque_cos + self.pos[1]
        self.thrustets_center[1][0] = + x2 * torque_cos - y2 * torque_sin + self.pos[0]
        self.thrustets_center[1][1] = + x2 * torque_sin + y2 * torque_cos + self.pos[1]
            
        # =========================
        # DEBUG
        # =========================
        self.debug_visuals = [
            [dx, dy],
            [thr_1_x, thr_1_y],
            [thr_2_x, thr_2_y],
        ]

        # =========================
        # SMOKE
        # =========================
        if not self.no_smoke:
            [part.physics_step(step_dt) for part in self.reactor_particles]

            for i in range(2):
                self.reactor_particles.append(Smoke(
                    self.thrustets_center[0][0] + math.cos(math.pi / 2 + math.radians(self.thrustets_rotations_global[0])) * self.size_perc(0.3), 
                    self.thrustets_center[0][1] + math.sin(math.pi / 2 + math.radians(self.thrustets_rotations_global[0])) * self.size_perc(0.3), 
                    (random.random() * 0.50 + 0.50 * self.thrusters_power[0]) * 13 / step_dt, 
                    [random.random() * 0.4 + 0.6 * math.cos(math.pi / 2 + math.radians(self.thrustets_rotations_global[0])), 
                    random.random() * 0.4 + 0.6 * math.sin(math.pi / 2 + math.radians(self.thrustets_rotations_global[0]))],
                    15,
                    10 * self.thrusters_power[0]
                ))
                
                self.reactor_particles.append(Smoke(
                    self.thrustets_center[1][0] + math.cos(math.pi / 2 + math.radians(self.thrustets_rotations_global[1])) * self.size_perc(0.3), 
                    self.thrustets_center[1][1] + math.sin(math.pi / 2 + math.radians(self.thrustets_rotations_global[1])) * self.size_perc(0.3), 
                    (random.random() * 0.50 + 0.50 * self.thrusters_power[1]) * 13 / step_dt, 
                    [random.random() * 0.4 + 0.6 * math.cos(math.pi / 2 + math.radians(self.thrustets_rotations_global[1])), 
                    random.random() * 0.4 + 0.6 * math.sin(math.pi / 2 + math.radians(self.thrustets_rotations_global[1]))],
                    15,
                    10 * self.thrusters_power[1]
                ))

            self.reactor_particles = [i for i in self.reactor_particles if i.alive]




class Smoke:
    def __init__(self, x, y, speed, dir, life_average_step=10, initial_radius=5):
        self.x = x
        self.y = y
        self.speed = speed
        self.dir = dir
        self.life_step = 0
        self.radius = initial_radius
        self.radius0 = initial_radius
        self.life_max_step = life_average_step + (life_average_step / 10) * random.random()
        self.orientation = random.random() * 360
        self.angular_speed = random.random() * 0.1


    def physics_step(self, step_dt=1):
        self.x += self.speed * self.dir[0] * step_dt
        self.y += self.speed * self.dir[1] * step_dt

        self.radius *= 1.15
        self.speed *= 0.95
        self.orientation += self.angular_speed * step_dt
        
        self.life_step += 1


    @property
    def alive(self):
        return self.life_step < self.life_max_step
