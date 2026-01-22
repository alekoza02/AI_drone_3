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
        font_path = pygame.font.match_font("dejavusansmono")
        self.font = pygame.font.Font(font_path, 18)
        

    def clear(self):
        self.screen.fill(self.bg_color)


    def update(self, fps=60):
        pygame.display.flip()
        self.clock.tick(fps)


    def render_polygon(self, points, color):    
        pygame.draw.polygon(self.screen, [x * 255 if x < 1 else 255 for x in color], points)


    def render_point(self, point, color, radius=10):
        pygame.draw.circle(self.screen, color, point, radius)
    

    def render_line(self, start, end,  color, width=10):
        pygame.draw.line(self.screen, color, start, end, width)


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


    def render_NN(self, pos, info):
        
        circles, links = info

        for circle in circles:
            color = [0, abs(circle['intensity']) / 2 * 255, 0]

            self.render_point([pos[0] + circle['x'], pos[1] + circle['y']], color, circle['radius'])
        
        for link in links:
            color = [0, link['intensity'] * 255, 255 - link['intensity'] * 255]
            self.render_line([pos[0] + link['start'][0], pos[1] + link['start'][1]], [pos[0] + link['end'][0], pos[1] + link['end'][1]], color, max(1, int(5 * link['intensity'])))

    def render_drone(self, info):
        
        for key, value in info.items():
            verteces = value['verteces']
            verteces = self.rotate_verteces(verteces, value['global_rotation'], value['global_position'])
            verteces = self.rotate_verteces(verteces, value['local_rotation'], value['local_position'])
            self.render_polygon(verteces, value['color'])

        # DEBUG VISUALIZATION OF ROTATION CENTERS
        # for key, value in info.items():
        #     self.render_point(value['global_position'], [0, 255, 0])
        #     self.render_point(value['local_position'], [0, 0, 255])


    def render_direction(self, pos, dir, scale=1):
        pygame.draw.circle(self.screen, [100, 100, 255], pos, 10)
        pygame.draw.line(self.screen, [100, 100, 255], pos, [pos[0] + dir[0] * scale, pos[1] + dir[1] * scale], 6)
        

    def render_text(self, text, antialias, color, pos):
        text_surface = self.font.render(text, antialias, color)
        self.screen.blit(text_surface, pos)