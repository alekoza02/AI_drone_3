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


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-train", type=str, default="0")
parser.add_argument("-resume", type=str, default="0")
parser.add_argument("-play", type=str, default="1")

args = parser.parse_args()

TRAIN = args.train.lower() == "1"
PLAY = args.play.lower() == "1"
RESUME = args.resume.lower() == "1"

force_no_render = True
visual_update = 4
dt_frame = 1
elapsed = 0
running = True

N_agents = 2 ** 10
max_training_steps = 1800
N_epoch = 2 ** 15

wavepoints = [[0.5 * W, 0.5 * H]]
wavepoints.extend([[random() * W, random() * H] for i in range(50)])

prev_best = -1e9

if not force_no_render:
    r: Renderer = Renderer(W, H)
else: 
    r = None

if RESUME:
    print("Resuming training...")
    arena = Arena(N_agents, N_epoch, max_training_steps, [W/2, H/2])
    agent = arena.load_best_NN('autosave.pkl')
    arena.populate_with_loaded_NN(agent)
else:
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


        if arena.current_step < arena.max_steps and not arena.force_new_epoch:
            arena.train_single_step()
            prev_best = max(prev_best, arena.best_prev_score)
        else:
            arena.current_epoch += 1
            if arena.current_epoch >= arena.n_epoch:
                arena.training_finished = True
                arena.save_best_NN('new.pkl')
            else:        
                arena.save_best_NN('autosave.pkl')
                arena.start_next_epoch()
                print("")
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
            # r.render_text(f" INPUTS MAX {arena.max_inputs[0]:.2f}, {arena.max_inputs[1]:.2f}, {arena.max_inputs[2]:.2f}, {arena.max_inputs[3]:.2f}, {arena.max_inputs[4]:.2f}, {arena.max_inputs[5]:.2f}, {arena.max_inputs[6]:.2f}", True, (255, 255, 255), (0, 50))
            # r.render_text(f" INPUTS MIN {arena.min_inputs[0]:.2f}, {arena.min_inputs[1]:.2f}, {arena.min_inputs[2]:.2f}, {arena.min_inputs[3]:.2f}, {arena.min_inputs[4]:.2f}, {arena.min_inputs[5]:.2f}, {arena.min_inputs[6]:.2f}", True, (255, 255, 255), (0, 100))
            # r.render_text(f" OUTPUTS MAX {arena.max_outputs[0]:.2f}, {arena.max_outputs[1]:.2f}, {arena.max_outputs[2]:.2f}, {arena.max_outputs[3]:.2f}", True, (255, 255, 255), (0, 150))
            # r.render_text(f" OUTPUTS MIN {arena.min_outputs[0]:.2f}, {arena.min_outputs[1]:.2f}, {arena.min_outputs[2]:.2f}, {arena.min_outputs[3]:.2f}", True, (255, 255, 255), (0, 200))
            r.update(0)

        elif elapsed > 100:
            elapsed = 0
            print(f"\rACTIVE {sum(arena.agents_mask):<5} | IA {(arena.timings_ia) / 1000000:7.0f}ms | PHYSICS {(arena.timings_physics) / 1000000:7.0f}ms | EPOCHS {arena.current_epoch:5} / {arena.n_epoch:5} | STEPS {arena.current_step:5} / {arena.max_steps:5} | BEST SCORE {prev_best:8.1f}", end="")

        stop_dt = perf_counter_ns()
        dt_frame = stop_dt - start_dt
        elapsed += dt_frame / 1e6



def reload_drone(arena, loaded_model):
    drone = Drone(W*0.5, H*0.5)
    agent = arena.load_best_NN(loaded_model)
    drone.no_smoke = False

    return drone, agent

if PLAY:

    loaded_model = 'autosave.pkl'
    # loaded_model = 'finished_training.pkl'
    # loaded_model = 'new.pkl'

    drone, agent = reload_drone(arena, loaded_model)

    if r is None:
        r: Renderer = Renderer(W, H)

    while running:
        
        start_dt = perf_counter_ns()

        for event in pygame.event.get():

            if event.type == pygame.MOUSEMOTION:
                drone.destination = event.pos
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    drone, agent = reload_drone(arena, loaded_model)

            if event.type == pygame.QUIT:
                running = False

        try:

            start_render = perf_counter_ns()
            r.clear()

            delta_ia, delta_physics = arena.raw_step(drone, agent, step_dt=1)

            r.render_drone(drone.get_drone_visual_info())
            r.render_NN([0, 2 * H / 3], agent.get_NN_visual_info(W / 3, H / 3))

            r.render_point(drone.destination, [255, 255, 0])
            stop_render = perf_counter_ns()
            
            r.render_text(f" IA {delta_ia / 1000:3.0f}us | PHYSICS {delta_physics / 1000:4.0f}us | RENDER {(stop_render - start_render) / 1000:5.0f}us | FPS {1 / (dt_frame * 1e-9):5.0f} | LOADED MODEL: {loaded_model}", True, (255, 255, 255), (0, 0))
            r.render_text(f" CURRENT OUTPUT: {drone.thrustets_rotations_local[0]:.2f}°, {drone.thrustets_rotations_local[1]:.2f}°, {drone.thrusters_power[0] * 100:.2f}%, {drone.thrusters_power[1] * 100:.2f}%", True, (255, 255, 255), (0, 50))
            
            r.update(60)

        except Exception as e:
            print(e)
            drone, agent = reload_drone(arena, loaded_model)

        stop_dt = perf_counter_ns()
        dt_frame = stop_dt - start_dt
        elapsed += dt_frame / 1e6

        if elapsed > 5000:
            elapsed = 0
            # load newest agent
            agent = arena.load_best_NN(loaded_model)

    pygame.quit()


if PROFILING:
    yappi.stop()
    func_stats = yappi.get_func_stats()
    func_stats.save('profilatore.prof', type='pstat') 