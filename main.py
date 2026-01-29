from arena import *
from neural_network import *
from small_renderer import *
from drone import *
from arena import *

from time import perf_counter_ns

PROFILING = False

if PROFILING:
    import yappi
    yappi.start()

W, H = 2000, 1000

TRAIN = 0
PLAY = 1

force_no_render = True
visual_update = 1
dt_frame = 1
elapsed = 0
running = True

N_agents = 2 ** 10
max_training_steps = 600
N_epoch = 2 ** 12

wavepoints = [[0.5 * W, 0.5 * H], [0.75 * W, 0.25 * H], [0.25 * W, 0.25 * H], [0.75 * W, 0.75 * H], [0.25 * W, 0.75 * H]]
# wavepoints = [[0.75 * W, 0.25 * H], [0.25 * W, 0.25 * H], [0.75 * W, 0.75 * H], [0.25 * W, 0.75 * H]]

prev_best = -1e9

if not force_no_render:
    r: Renderer = Renderer(W, H)
else: 
    r = None

arena = Arena(N_agents, N_epoch, max_training_steps, [W/2, H/2])
arena.set_simulation_world_size([W, H])
arena.set_goal(wavepoints)

if TRAIN:

    while not arena.training_finished:
        
        start_dt = perf_counter_ns()

        if not force_no_render:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    arena.training_finished = True
                    arena.save_best_NN('autosave.pkl')

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        arena = Arena(N_agents, N_epoch, max_training_steps, [W/2, H/2])
                        arena.set_simulation_world_size([W, H])
                        arena.set_goal(wavepoints)
        
        if arena.current_step < arena.max_steps and not arena.force_new_epoch:
            arena.train_single_step()
            prev_best = max(prev_best, arena.best_prev_score)
        else:
            arena.current_epoch += 1
            if arena.current_epoch >= arena.n_epoch:
                arena.training_finished = True
                arena.save_best_NN('finished_training.pkl')
            else:        
                arena.start_next_epoch()
                arena.train_single_step()
                prev_best = max(prev_best, arena.best_prev_score)

        start_render = perf_counter_ns()
        
        if not force_no_render:
            r.clear()
        
        # WHOLE SWARM
        if arena.current_epoch % visual_update == 0 and not force_no_render:
            for index, drone in enumerate(arena.agents_drone):
                r.render_point(drone.destination, [255 * (index / arena.n_agents), 255 * (1 - index / arena.n_agents), 0])
            for drone in arena.agents_drone:
                r.render_drone(drone.get_drone_visual_info())
                r.render_direction(drone.pos, drone.debug_visuals[0], scale=10)
                r.render_direction(drone.thrustets_center[0], drone.debug_visuals[1], scale=100)
                r.render_direction(drone.thrustets_center[1], drone.debug_visuals[2], scale=100)
            
            r.render_NN([0, 3 * H / 4], arena.agents_NN[arena.best_scores_indices[0]].get_NN_visual_info(W / 4, H / 4))

        stop_render = perf_counter_ns()

        if not force_no_render:
            r.render_text(f" ACTIVE {sum(arena.agents_mask):<5} | FPS {1 / (dt_frame * 1e-9):5.0f} | EPOCHS {arena.current_epoch} / {arena.n_epoch} | STEPS {arena.current_step} / {arena.max_steps} | BEST SCORE {prev_best:.2f} | CURRENT BEST SCORE {arena.best_prev_score:.2f}", True, (255, 255, 255), (0, 0))
            r.render_text(f" INPUTS MAX {arena.max_inputs[0]:.2f}, {arena.max_inputs[1]:.2f}, {arena.max_inputs[2]:.2f}, {arena.max_inputs[3]:.2f}, {arena.max_inputs[4]:.2f}, {arena.max_inputs[5]:.2f}, {arena.max_inputs[6]:.2f}", True, (255, 255, 255), (0, 50))
            r.render_text(f" INPUTS MIN {arena.min_inputs[0]:.2f}, {arena.min_inputs[1]:.2f}, {arena.min_inputs[2]:.2f}, {arena.min_inputs[3]:.2f}, {arena.min_inputs[4]:.2f}, {arena.min_inputs[5]:.2f}, {arena.min_inputs[6]:.2f}", True, (255, 255, 255), (0, 100))
            r.render_text(f" OUTPUTS MAX {arena.max_outputs[0]:.2f}, {arena.max_outputs[1]:.2f}, {arena.max_outputs[2]:.2f}, {arena.max_outputs[3]:.2f}", True, (255, 255, 255), (0, 150))
            r.render_text(f" OUTPUTS MIN {arena.min_outputs[0]:.2f}, {arena.min_outputs[1]:.2f}, {arena.min_outputs[2]:.2f}, {arena.min_outputs[3]:.2f}", True, (255, 255, 255), (0, 200))
            r.update(0)

        elif elapsed > 1000:
            elapsed = 0
            print(f"\r ACTIVE {sum(arena.agents_mask):<5} | IA {(arena.timings_ia) / 1000000:3.0f}ms | PHYSICS {(arena.timings_physics) / 1000000:4.0f}ms | RENDER {(stop_render - start_render) / 1000:5.0f}us | FPS {1 / (dt_frame * 1e-9):5.0f} | EPOCHS {arena.current_epoch} / {arena.n_epoch} | STEPS {arena.current_step} / {arena.max_steps} | BEST SCORE {prev_best:.1f}               ", end="")

        stop_dt = perf_counter_ns()
        dt_frame = stop_dt - start_dt
        elapsed += dt_frame / 1e6


if PLAY:
    drone = Drone(W*0.5, H*0.5)
    agent = arena.load_best_NN('finished_training.pkl')
    # agent = arena.load_best_NN('autosave.pkl')

    if r is None:
        r: Renderer = Renderer(W, H)

    while running:
        
        start_dt = perf_counter_ns()

        for event in pygame.event.get():

            if event.type == pygame.MOUSEMOTION:
                drone.destination = event.pos

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

        r.render_point(drone.destination, [255, 255, 0])
        stop_render = perf_counter_ns()
        
        r.render_text(f" IA {delta_ia / 1000:3.0f}us | PHYSICS {delta_physics / 1000:4.0f}us | RENDER {(stop_render - start_render) / 1000:5.0f}us | FPS {1 / (dt_frame * 1e-9):5.0f}", True, (255, 255, 255), (0, 0))
        
        r.update(60)

        stop_dt = perf_counter_ns()
        dt_frame = stop_dt - start_dt

    pygame.quit()

if PROFILING:
    yappi.stop()
    func_stats = yappi.get_func_stats()
    func_stats.save('profilatore.prof', type='pstat') 