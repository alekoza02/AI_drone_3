import pygame
import math

class Renderer:
    def __init__(self, width, height, bg_color=(30, 30, 30), title="Renderer"):
        pygame.init()
        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        self.bg_color = bg_color
        self.clock = pygame.time.Clock()


    def clear(self):
        self.screen.fill(self.bg_color)


    def update(self, fps=60):
        pygame.display.flip()
        self.clock.tick(fps)


    def render_polygon(self, points, color):    
        pygame.draw.polygon(self.screen, color, points)


    def render_point(self, point, color):
        pygame.draw.circle(self.screen, color, point, 10)


    def rotate_verteces(self, points, angle, rotation_center):
        
        if angle == 0:
            return points

        rotated = []

        angle_rad = math.radians(angle)        
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        for x, y in points:
            
            x -= rotation_center[0]
            y -= rotation_center[1]

            rx = x * cos_a - y * sin_a + rotation_center[0]
            ry = x * sin_a + y * cos_a + rotation_center[1]
            rotated.append((rx, ry))

        return rotated


    def render_drone(self, info):
        
        for key, value in info.items():
            verteces = value['verteces']
            verteces = self.rotate_verteces(verteces, value['global_rotation'], value['global_position'])
            value['local_position'] = self.rotate_verteces([value['local_position']], value['global_rotation'], value['global_position'])[0]
            verteces = self.rotate_verteces(verteces, value['local_rotation'], value['local_position'])
            self.render_polygon(verteces, value['color'])

        # DEBUG VISUALIZATION OF ROTATION CENTERS
        # for key, value in info.items():
        #     self.render_point(value['global_position'], [0, 255, 0])
        #     self.render_point(value['local_position'], [0, 0, 255])
