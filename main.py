from arena import *
from neural_network import *
from small_renderer import *
from drone import *
from arena import *

from time import perf_counter_ns

W, H = 2000, 1000

N_agents = 1000
max_training_steps = 2000
N_epoch = 1

r: Renderer = Renderer(W, H)

arena = Arena(N_agents, 1, max_training_steps, [W/2, H/2])
arena.set_simulation_world_size([W, H])
arena.set_goal([0.86 * W, 0.13 * H])

running = True
dt_frame = 1

while running:
    
    start_dt = perf_counter_ns()

    rotation1 = 0
    rotation2 = 0

    for event in pygame.event.get():

        # if event.type == pygame.MOUSEMOTION:
        #     arena.destination = event.pos

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                arena = Arena(N_agents, 1, max_training_steps, [W/2, H/2])
                arena.set_simulation_world_size([W, H])
                arena.set_goal([0.86 * W, 0.13 * H])
    
    if arena.current_step < arena.max_steps:
        arena.train_single_step()
    else:
        arena.start_next_epoch()

    start_render = perf_counter_ns()
    r.clear()

    r.render_drone(arena.agents_drone[arena.best_scores_indices[0]].get_drone_visual_info())
    r.render_direction(arena.agents_drone[arena.best_scores_indices[0]].pos, arena.agents_drone[arena.best_scores_indices[0]].debug_visuals[0], scale=10)
    r.render_direction(arena.agents_drone[arena.best_scores_indices[0]].thrustets_center[0], arena.agents_drone[arena.best_scores_indices[0]].debug_visuals[1], scale=100)
    r.render_direction(arena.agents_drone[arena.best_scores_indices[0]].thrustets_center[1], arena.agents_drone[arena.best_scores_indices[0]].debug_visuals[2], scale=100)
    
    r.render_NN([0, H / 2], arena.agents_NN[arena.best_scores_indices[0]].get_NN_visual_info(W / 2, H / 2))

    r.render_point(arena.destination, [255, 255, 0])
    stop_render = perf_counter_ns()
    
    r.render_text(f" ACTIVE {sum(arena.agents_mask)} | IA {(arena.timings_ia / arena.timings_sample) / 1000:3.0f}us | PHYSICS {(arena.timings_physics / arena.timings_sample) / 1000:4.0f}us | RENDER {(stop_render - start_render) / 1000:5.0f}us | FPS {1 / (dt_frame * 1e-9):5.0f} | STEPS {arena.current_step} / {arena.max_steps} | BEST SCORE {sum(arena.agents_scores[arena.best_scores_indices[0]]):.4f}", True, (255, 255, 255), (0, 0))

    r.update(0)

    stop_dt = perf_counter_ns()
    dt_frame = stop_dt - start_dt

pygame.quit()
