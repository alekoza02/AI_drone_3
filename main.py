from arena import *
from neural_network import *
from small_renderer import *
from drone import *

r = Renderer(800, 600)
d = Drone(400, 300)

running = True
angle = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    d.set_orientation(angle, [-angle, -angle])
    
    r.clear()
    r.render_drone(d.get_drone_visual_info())
    r.update()

    angle += 1

pygame.quit()
