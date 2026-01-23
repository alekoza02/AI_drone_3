from arena import *
from neural_network import *
from small_renderer import *
from drone import *
from arena import *

from time import perf_counter_ns

W, H = 2000, 1000

N_agents = 512
max_training_steps = 1024
N_epoch = 2 ** 4
N_evolutions = 32

r: Renderer = Renderer(W, H)

arena = Arena(N_agents, N_epoch, max_training_steps, N_evolutions, [W/2, H/2])
arena.set_simulation_world_size([W, H])
arena.set_goal([0.75 * W, 0.75 * H])

dt_frame = 1
running = True

TRAIN = 1
PLAY = 1

if TRAIN:

    while not arena.training_finished:
        
        start_dt = perf_counter_ns()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = arena.training_finished

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    arena = Arena(N_agents, N_epoch, max_training_steps, N_evolutions, [W/2, H/2])
                    arena.set_simulation_world_size([W, H])
                    arena.set_goal([0.86 * W, 0.13 * H])
        
        if arena.current_step < arena.max_steps and not arena.force_new_epoch:
            arena.train_single_step()
        else:
            arena.current_epoch += 1
            if arena.current_epoch >= arena.n_epoch:
                arena.training_finished = True
                arena.save_best_NN('new_best_neuron.pkl')
            else:        
                arena.start_next_epoch()
                arena.train_single_step()

        start_render = perf_counter_ns()
        r.clear()
        
        # WHOLE SWARM
        if arena.current_epoch % 32 == 0:
            for drone in arena.agents_drone:
                r.render_drone(drone.get_drone_visual_info())
                r.render_direction(drone.pos, drone.debug_visuals[0], scale=10)
                r.render_direction(drone.thrustets_center[0], drone.debug_visuals[1], scale=100)
                r.render_direction(drone.thrustets_center[1], drone.debug_visuals[2], scale=100)
            
            r.render_NN([0, 3 * H / 4], arena.agents_NN[arena.best_scores_indices[0]].get_NN_visual_info(W / 4, H / 4))

            r.render_point(arena.destination, [255, 255, 0])
        
        stop_render = perf_counter_ns()

        r.render_text(f" ACTIVE {sum(arena.agents_mask):<5} | IA {(arena.timings_ia / arena.timings_sample) / 1000:3.0f}us | PHYSICS {(arena.timings_physics / arena.timings_sample) / 1000:4.0f}us | RENDER {(stop_render - start_render) / 1000:5.0f}us | FPS {1 / (dt_frame * 1e-9):5.0f} | EPOCHS {arena.current_epoch} / {arena.n_epoch} | STEPS {arena.current_step} / {arena.max_steps} | BEST SCORE {arena.agents_scores[arena.best_scores_indices[0]]:.4f}", True, (255, 255, 255), (0, 0))
        r.update(0)

        stop_dt = perf_counter_ns()
        dt_frame = stop_dt - start_dt


if PLAY:
    drone = Drone(W*0.5, H*0.5)
    agent = arena.load_best_NN('new_best_neuron.pkl')

    while running:
        
        start_dt = perf_counter_ns()

        for event in pygame.event.get():

            if event.type == pygame.MOUSEMOTION:
                arena.destination = event.pos

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    drone = Drone(W*0.5, H*0.5)

            if event.type == pygame.QUIT:
                running = False


        start_render = perf_counter_ns()
        r.clear()

        delta_ia, delta_physics = arena.raw_step(drone, agent)

        r.render_drone(drone.get_drone_visual_info())
        r.render_NN([0, 3 * H / 4], agent.get_NN_visual_info(W / 4, H / 4))

        r.render_point(arena.destination, [255, 255, 0])
        stop_render = perf_counter_ns()
        
        r.render_text(f" IA {delta_ia / 1000:3.0f}us | PHYSICS {delta_physics / 1000:4.0f}us | RENDER {(stop_render - start_render) / 1000:5.0f}us | FPS {1 / (dt_frame * 1e-9):5.0f}", True, (255, 255, 255), (0, 0))
        
        r.update(60)

        stop_dt = perf_counter_ns()
        dt_frame = stop_dt - start_dt

    pygame.quit()