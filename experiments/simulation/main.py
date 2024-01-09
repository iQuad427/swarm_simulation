import pickle
import random
import threading
import time

import numpy as np
import pygame
from matplotlib import pyplot as plt
from numpy import linspace
from sklearn.manifold import MDS
from tqdm import tqdm

from agent import Agent
from experiments.simulation.observer import reconstruct_distance_matrix
from experiments.utils.plot import plot_on_top
from experiments.utils.rotation import rotate_and_translate

# Define the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)
RED = (255, 0, 0)

# Define the screen size
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
frame_rate = 120

# Define the arena
arena_width = SCREEN_WIDTH
arena_height = SCREEN_HEIGHT

# Show the arena
arena = pygame.Rect(
    (SCREEN_WIDTH - arena_width) / 2,
    (SCREEN_HEIGHT - arena_height) / 2,
    arena_width,
    arena_height,
)

# Define the agents
radius = 5
speed = 10
agents = [
    Agent(
        id=i,
        # Place randomly in the arena
        x=random.randint(0 + radius, arena_width - radius),
        y=random.randint(0 + radius, arena_height - radius),
        radius=radius,
        color=RED,
        speed=speed / frame_rate,
    ) for i in range(10)
]

# Define the clock
clock = pygame.time.Clock()

# Define the running state
running = True


def simulation_step():
    for agent in agents:
        agent.random_move(arena_width, arena_height)
        agent.collide(agents)


def run_simulation(frame_rate, plot=False):
    global running, agents

    # Define the screen
    if plot:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simulation")
    else:
        screen = None

    # Main loop
    while running:
        if plot:
            # Handle events
            for event in pygame.event.get():
                # Quit the game
                if event.type == pygame.QUIT:
                    running = False

            # Fill the screen
            screen.fill(BLACK)
            pygame.draw.rect(screen, WHITE, arena)

        simulation_step()

        # Show the agents
        if plot:
            for agent in agents:
                pygame.draw.circle(screen, agent.color, (agent.x, agent.y), agent.radius)

            # Update the screen
            pygame.display.flip()

        # Wait for the next frame
        clock.tick(frame_rate)


def make_experiment(sensor_noise: int = 0, plot=False):
    global agents

    measures = []
    for agent in agents:
        measures.append(agent.measure_distances(agents, noise=sensor_noise))

    discount_matrix, distance_matrix = reconstruct_distance_matrix(measures)

    # Perform classical multidimensional scaling
    mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, metric=True)
    mds_coors = mds.fit_transform(distance_matrix)

    # Added minus sign to flip the y-axis due to the different coordinate system of pygame and matplotlib
    points = np.array([[agent.x, -agent.y] for agent in agents])

    mds_coors = rotate_and_translate(points, mds_coors)

    # Plot the results
    if plot:
        plt.xlim(0, arena_width)
        plt.ylim(-arena_height, 0)
        plot_on_top(points, mds_coors)

    # Compute the error between the original and the reconstructed coordinates
    error = np.linalg.norm(points - mds_coors) / len(points)

    return error


def run_experiment(refresh_time=1, sensor_noise=0):
    global running, agents

    while running:
        make_experiment(sensor_noise=sensor_noise)
        time.sleep(refresh_time)


if __name__ == '__main__':
    independent_runs = 10

    min_noise = 0
    max_noise = 200
    granularity = 11
    noises = linspace(min_noise, max_noise, granularity)  # 0, 5, 10, ..., 100 (per 5 centimeters)

    experiment_duration = 1  # in seconds

    all_results = {noise: [] for noise in noises}

    for i, noise in enumerate(noises):
        for _ in tqdm(
                range(independent_runs),
                desc="Noise: " + str(noise) + " cm, Fraction: " + str(i + 1) + "/" + str(len(noises))
        ):
            start = time.time()

            # Run the simulation
            simulation_thread = threading.Thread(target=run_simulation, args=(frame_rate,))
            simulation_thread.start()

            results = []

            while time.time() - start < experiment_duration:
                result = make_experiment(sensor_noise=noise, plot=False)
                results.append(result)

            all_results[noise].append(np.mean(results))

    # Save the results dictionary
    with open(f'results/results_{min_noise}_{max_noise}_{granularity}', 'wb') as f:
        f.write(pickle.dumps(all_results))

    # Stop the simulation
    running = False

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.xlabel("Sensor noise (cm)")
    plt.ylabel("Error (cm)")
    plt.title("Error vs. sensor noise")
    plt.plot(noises, [np.mean(all_results[noise]) for noise in noises])
    # Add error bars
    plt.errorbar(
        noises,
        [np.mean(all_results[noise]) for noise in noises],
        yerr=[np.std(all_results[noise]) for noise in noises],
        fmt='o'
    )
    plt.show()

