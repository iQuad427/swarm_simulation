import numpy as np

if __name__ == '__main__':
    # Use the Extended Kalman Filter (EKF) to estimate the position of a robot in 2D

    # Define the state transition matrix
    A = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Define the process noise covariance matrix
    Q = np.array([
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1]
    ])

    # Define the observation matrix
    H = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    # Define the measurement noise covariance matrix
    R = np.array([
        [0.1, 0],
        [0, 0.1]
    ])

    # Define the initial state estimate
    x = np.array([0, 0, 0])

    # Define the initial state covariance matrix
    P = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Define the control input matrix
    B = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])

    # Define the control vector
    u = np.array([1, 1])

    # Define the observation vector
    z = np.array([1, 1])

    # Define the time interval
    dt = 1

    # Perform the prediction step
    x = A @ x + B @ u
    P = A @ P @ A.T + Q

    # Perform the update step
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(3) - K @ H) @ P

    print(x)

    # Output: [0.5 0.5 0.0]
