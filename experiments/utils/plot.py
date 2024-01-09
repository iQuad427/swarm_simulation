import matplotlib.pyplot as plt


def plot_side_by_side(points, mds_coors):
    # Plot the results side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].scatter(points[:, 0], points[:, 1], s=50, alpha=0.5)
    for i, point in enumerate(points):
        axs[0].annotate(str(i), (point[0], point[1]))

    axs[1].scatter(mds_coors[:, 0], mds_coors[:, 1], s=50, alpha=0.5)
    for i, point in enumerate(mds_coors):
        axs[1].annotate(str(i), (point[0], point[1]))

    plt.show()


def plot_on_top(points, mds_coors):
    # Plot the results side by side
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.scatter(points[:, 0], points[:, 1], color='red', s=50, alpha=0.5)
    for i, point in enumerate(points):
        ax.annotate(str(i), (point[0], point[1]))

    ax.scatter(mds_coors[:, 0], mds_coors[:, 1], color='blue', s=50, alpha=0.5)
    for i, point in enumerate(mds_coors):
        ax.annotate(str(i), (point[0], point[1]))

    # Legend for the plot
    ax.legend(['Original', 'MDS'], loc='best')

    plt.show()
