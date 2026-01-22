from arena import *
from neural_network import *
from small_renderer import *
from drone import *

from math import sin, cos, radians
from time import perf_counter_ns

W, H = 2000, 1000

r: Renderer = Renderer(W, H)
d: Drone = Drone(W / 2, H / 2)
n: NeuralNetwork = NeuralNetwork()
d.thrusters_power = [0.24, 0.24]
destination_point = [0.86 * W, 0.13 * H]

running = True
dt_frame = 1

while running:
    
    start_dt = perf_counter_ns()

    rotation1 = 0
    rotation2 = 0

    for event in pygame.event.get():

        if event.type == pygame.MOUSEMOTION:
            destination_point = event.pos

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                d: Drone = Drone(W / 2, H / 2)
                n: NeuralNetwork = NeuralNetwork()

    if abs((destination_point[0] - d.pos[0]) / W) > 1 or abs((destination_point[1] - d.pos[1]) / H) > 1:
        d: Drone = Drone(W / 2, H / 2)
        n: NeuralNetwork = NeuralNetwork()



    inputs = [
        (destination_point[0] - d.pos[0]) / W,   # 1) to target X        
        (destination_point[1] - d.pos[1]) / H,   # 2) to target Y        
        d.speed[0] / 1200,                          # 3) speed X        
        d.speed[1] / 1200,                          # 4) speed Y        
        cos(radians(d.rotation)),                   # 5) cos angle        
        sin(radians(d.rotation)),                   # 6) sin angle
        d.angular_velocity,                         # 7) angular speed
    ]


    start_ia = perf_counter_ns()
    output = n.IA_step(inputs)
    stop_ia = perf_counter_ns()
    
    # print("INPUTS: ", inputs)
    # print("OUTPUTS: ", output)
    
    d.thrusters_power = [output[0], output[2]]
    d.thrustets_rotations_local = [output[1], output[3]]

    start_physics = perf_counter_ns()    
    dirs = d.physics_simulation_step()
    stop_physics = perf_counter_ns()    

    start_render = perf_counter_ns()
    r.clear()
    r.render_drone(d.get_drone_visual_info())
    r.render_NN([0, H / 2], n.get_NN_visual_info(W / 2, H / 2))

    r.render_point(destination_point, [255, 255, 0])
    r.render_direction(d.pos, dirs[0], scale=10)
    r.render_direction(d.thrustets_center[0], dirs[1], scale=100)
    r.render_direction(d.thrustets_center[1], dirs[2], scale=100)
    stop_render = perf_counter_ns()
    
    r.render_text(f"IA {(stop_ia - start_ia) / 1000:3.0f}ms | PHYSICS {(stop_physics - start_physics) / 1000:4.0f}ms | RENDER {(stop_render - start_render) / 1000:5.0f}ms | FPS {1 / dt_frame:5.0f}", True, (255, 255, 255), (0, 0))

    r.update()

    stop_dt = perf_counter_ns()
    dt_frame = stop_dt - start_dt

pygame.quit()
