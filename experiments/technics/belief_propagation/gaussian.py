import numpy as np
from scipy.stats import multivariate_normal


class GaussianBeliefPropagation:
    def __init__(self, num_robots, spatial_dimensions):
        self.num_robots = num_robots
        self.spatial_dimensions = spatial_dimensions
        self.beliefs = [None] * num_robots
        self.messages = [[None] * num_robots for _ in range(num_robots)]

    def initialize_beliefs(self):
        # Initialize beliefs with zero mean and identity covariance matrix
        for i in range(self.num_robots):
            self.beliefs[i] = multivariate_normal(mean=np.zeros(self.spatial_dimensions),
                                                  cov=np.eye(self.spatial_dimensions))

    def update_messages(self, range_measurements):
        # Update messages based on range measurements
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if i != j:
                    # Update message from robot i to robot j
                    precision = np.eye(self.spatial_dimensions)
                    precision += np.outer(range_measurements[i][j], range_measurements[i][j])
                    covariance = np.linalg.inv(precision)
                    self.messages[i][j] = multivariate_normal(mean=self.beliefs[i].mean, cov=covariance)

    # def update_beliefs(self):
    #     # Update beliefs based on incoming messages
    #     for i in range(self.num_robots):
    #         precision_sum = np.zeros((self.spatial_dimensions, self.spatial_dimensions))
    #         mean_sum = np.zeros(self.spatial_dimensions)
    #         for j in range(self.num_robots):
    #             if i != j:
    #                 precision_sum += np.linalg.inv(self.messages[j][i].cov)
    #                 mean_sum += np.dot(np.linalg.inv(self.messages[j][i].cov), self.messages[j][i].mean)
    #
    #         self.beliefs[i] = multivariate_normal(
    #             mean=np.dot(np.linalg.inv(precision_sum), mean_sum),
    #             cov=np.linalg.inv(precision_sum)
    #         )

    # def update_beliefs(self):
    #     # Update beliefs based on incoming messages
    #     for i in range(self.num_robots):
    #         mean_sum = np.zeros(self.spatial_dimensions)
    #         covariance_sum = np.zeros((self.spatial_dimensions, self.spatial_dimensions))
    #         for j in range(self.num_robots):
    #             if i != j:
    #                 mean_sum += self.messages[j][i].mean
    #                 covariance_sum += self.messages[j][i].cov
    #
    #         covariance_sum_inv = np.linalg.inv(covariance_sum)
    #
    #         self.beliefs[i] = multivariate_normal(
    #             mean=np.dot(covariance_sum_inv, mean_sum),
    #             cov=covariance_sum_inv
    #         )

    def update_beliefs(self):
        # Update beliefs based on incoming messages
        for i in range(self.num_robots):
            mean_sum = np.zeros(self.spatial_dimensions)
            covariance_sum = np.zeros((self.spatial_dimensions, self.spatial_dimensions))
            for j in range(self.num_robots):
                if i != j:
                    # Weighted sum of means and covariances based on precision matrices
                    precision_ji = np.linalg.inv(self.messages[j][i].cov)
                    mean_sum += np.dot(precision_ji, self.messages[j][i].mean)
                    covariance_sum += precision_ji
            covariance = np.linalg.inv(covariance_sum)
            mean = np.dot(covariance, mean_sum)
            self.beliefs[i] = multivariate_normal(mean=mean, cov=covariance)

    def infer_relative_configuration(self):
        # Infer relative configuration from final beliefs
        relative_config = []
        for i in range(self.num_robots):
            relative_config.append(self.beliefs[i].mean)

        return relative_config


# Example usage
if __name__ == "__main__":
    num_robots = 3
    spatial_dimensions = 2  # Assuming 2D environment
    # Simulated range measurements (symmetric matrix)
    range_measurements = np.array([[0, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 0]])

    gbp = GaussianBeliefPropagation(num_robots, spatial_dimensions)
    gbp.initialize_beliefs()

    # Perform belief propagation
    max_iterations = 10
    for _ in range(max_iterations):
        gbp.update_messages(range_measurements)
        gbp.update_beliefs()

    # Infer relative configuration
    relative_configuration = gbp.infer_relative_configuration()
    print("Inferred relative configuration:")
    for i in range(num_robots):
        print(f"Robot {i + 1}: {relative_configuration[i]}")
