class Drone:
    def __init__(self, x, y):
        self.pos = [x, y]
        self.rotation = 0

        self.thrusters_power = [255, 255]
        self.thrustets_rotations = [0, 0]
        
        self.angular_velocity = 0
        self.velocity = [0, 0]

        self.color_property_value = 255

        # -----------------
        self.size = 100

    
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
                'color' : [self.color_property_value, 0, 0],
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
                'color' : [self.color_property_value, 0, self.thrusters_power[0]],
                'global_rotation' : self.rotation,
                'local_rotation' : self.thrustets_rotations[0],
                'global_position' : self.pos,
                'local_position' : [self.pos[0] - self.size_perc(0.7), self.pos[1]], 
            },
            'thr_2' : {
                'verteces' : [
                    [self.pos[0] + self.size_perc(0.5) + self.size_perc(0.1) + self.size_perc(0.2), self.pos[1] - self.size_perc(0.3)],
                    [self.pos[0] + self.size_perc(0.5) + self.size_perc(0.1) + self.size_perc(0.2), self.pos[1] + self.size_perc(0.3)],
                    [self.pos[0] + self.size_perc(0.5) + self.size_perc(0.1), self.pos[1] + self.size_perc(0.3)],
                    [self.pos[0] + self.size_perc(0.5) + self.size_perc(0.1), self.pos[1] - self.size_perc(0.3)],
                ],
                'color' : [self.color_property_value, 0, self.thrusters_power[1]],
                'global_rotation' : self.rotation,
                'local_rotation' : self.thrustets_rotations[1],
                'global_position' : self.pos,
                'local_position' : [self.pos[0] + self.size_perc(0.7), self.pos[1]], 
            }
        }
    

    def set_orientation(self, corp_rotation, thr_rotation):
        self.rotation = corp_rotation
        self.thrustets_rotations = thr_rotation

    
    def set_position(self, position):
        self.pos = position