import pickle

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    with open("results/results_0_200_11", "rb") as f:
        results = f.read()

    all_results = pickle.loads(results)

    print(all_results)

    noises = list(all_results.keys())

    # Plot 2 different subplots
    # 1. Error vs. sensor noise
    # 2. Error over time
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Plot the results
    axs[0].set_xlabel("Sensor noise (cm)")
    axs[0].set_ylabel("Error (cm)")
    axs[0].set_title("Error vs. sensor noise")
    axs[0].plot(noises, [np.mean(all_results[noise]) for noise in noises])
    # Add confidence interval
    axs[0].fill_between(
        noises,
        [np.mean(all_results[noise]) - np.std(all_results[noise]) for noise in noises],
        [np.mean(all_results[noise]) + np.std(all_results[noise]) for noise in noises],
        alpha=0.2
    )
    # Add error bars
    axs[0].errorbar(
        noises,
        [np.mean(all_results[noise]) for noise in noises],
        yerr=[np.std(all_results[noise]) for noise in noises],
        fmt='o'
    )

    # Plot error error over time
    axs[1].set_xlabel("Experiment progression (%)")
    axs[1].set_ylabel("Error (cm)")
    axs[1].set_title("Error over time")
    for noise in noises:
        axs[1].plot(
            np.linspace(0, 1, len(all_results[noise])),
            all_results[noise],
            label=str(noise) + " cm"
        )
        results = all_results[noise]
        # Add confidence interval
        axs[1].fill_between(
            np.linspace(0, 1, len(results)),
            [np.mean(results) - np.std(results)] * len(results),
            [np.mean(results) + np.std(results)] * len(results),
            alpha=0.2
        )
    plt.legend()
    plt.show()

