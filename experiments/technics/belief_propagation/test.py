import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # List of 4 robots positions in 2D
    # - 4 bimodal gaussian distributions centered in the corners of a square
    # - Each distribution has a mean and a covariance matrix
    positions = [
        (np.array([0, 0]), np.array([[0.1, 0], [0, 0.1]])),
        (np.array([0, 1]), np.array([[0.1, 0], [0, 0.1]])),
        (np.array([1, 0]), np.array([[0.1, 0], [0, 0.1]])),
        (np.array([1, 1]), np.array([[0.1, 0], [0, 0.1]]))
    ]

    # Estimated positions are points drawn from the bimodal gaussian distributions
    estimated_positions = [
        np.random.multivariate_normal(positions[i][0], positions[i][1]) for i in range(4)
    ]

    for i in range(4):
        # Plot the positions of the robots as the mean of the bimodal gaussian distributions
        plt.scatter(positions[i][0][0], positions[i][0][1], color='tab:blue')

        # Plot the estimated positions of the robots
        plt.scatter(estimated_positions[i][0], estimated_positions[i][1], color='tab:orange')

        # Plot the covariance matrix as an ellipse
        cov = positions[i][1]
        mean = positions[i][0]
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = plt.matplotlib.patches.Ellipse(xy=mean, width=lambda_[0]*2, height=lambda_[1]*2, angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor('black')
        plt.gca().add_artist(ell)


    # Equal ratio for the axes
    plt.axis('equal')

    # Show the plot
    plt.show()
