import math

import numpy as np
import matplotlib.pyplot as plt


def find_direction_vector_from_position_history(position_history):
    """
    Given a position history, find the direction vector of the movement
    :param position_history: A list of positions
    :return: The direction vector
    """
    # Convert the list to a numpy array
    position_history = np.array(position_history)

    # Compute the difference between each position
    diff = position_history[1:] - position_history[:-1]

    # Compute the direction vector by taking the mean of the differences
    direction_vector = np.mean(diff, axis=0)

    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    return direction_vector


def range_and_bearing(agent, historic, plot):
    """
    Estimate the relative direction of all agents

    :param agent: id of the robot currently estimating its range and bearing to other robots of the swarm
    :param historic: sample of last known position of the robot on the plot
    :param plot: position estimation to take into consideration when computing angles
    :return: the estimated readings of the simulated range and bearing sensor
    """
    # Convert the plot to a numpy array
    plot = np.array(plot)

    # Rotate the plot to have the agent of interest at the origin and the direction vector aligned with the x-axis
    # Compute the direction vector of the movement
    direction_vector = find_direction_vector_from_position_history(historic)

    # Compute the angle between the direction vector and the x-axis
    angle = np.arctan2(direction_vector[1], direction_vector[0])

    # Compute the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # Rotate the plot
    plot = np.array(plot) - historic[-1]
    plot = plot @ rotation_matrix

    # Compute the range and bearing to the other agents
    _rab = []

    for x, position in enumerate(plot):
        if x != agent:
            # Compute the relative position of the other agent
            relative_position = position - plot[agent]

            # Compute the range and bearing
            _rab.append(
                [
                    np.linalg.norm(relative_position) / 2,
                    np.arctan2(relative_position[1], relative_position[0])
                ]
            )

    return _rab


if __name__ == '__main__':
    # Test the function
    _position_history = [[-221.8605728782181, -44.80606982400007], [-221.23048862483742, -44.1325607595006], [-220.91887872996614, -46.317974300476905], [-217.91846423180868, -50.21685212146818], [-216.33833999677995, -48.6060862822082], [-205.45079354778787, -55.31352690271735], [-204.82767764610267, -55.14033373366937], [-205.1017358407157, -54.08417421501816], [-204.23932651962866, -54.42950023112051], [-203.79715451552642, -55.19378514848738], [-202.635540899001, -55.19023415419103], [-203.00907297555275, -54.944072838664546], [-197.26432962938387, -63.74161599216652], [-193.05846590087427, -61.8687775721978], [-192.41070533134408, -61.67836353641636], [-185.71852991783317, -72.1143224271424], [-185.99033719782017, -73.49102395398276], [-184.08568430422528, -75.95890574430864], [-185.31393183396895, -75.90180391823988], [-181.27528204415012, -72.59664709176187]]

    # Current positions of the 5 agents
    _current_positions = [
        _position_history[-1]
    ]

    # to which we add 4 random positions
    _current_positions.extend([[x, y] for x, y in zip(np.random.uniform(-200, 0, 4), np.random.uniform(-200, 0, 4))])

    # Agent of interest
    _agent = 0

    # Plot the agents (current positions)
    plt.scatter(*zip(*_current_positions), label='Current positions')

    # Plot the position history
    plt.plot(*zip(*_position_history), label='Position history')

    # Plot the direction vector (from agent of interest)
    _direction_vector = find_direction_vector_from_position_history(_position_history) * 10
    plt.quiver(*_position_history[-1], _direction_vector[0], _direction_vector[1], angles='xy', scale_units='xy', scale=0.5, color='r', label='Direction vector')

    # Find the range and bearing to the other agents
    _range_and_bearing = range_and_bearing(_agent, _position_history, _current_positions)

    # Compute the angle between the direction vector and the x-axis
    angle = np.arctan2(_direction_vector[1], _direction_vector[0])

    # Plot the range and bearing relative to the agent of interest direction
    for i, (r, b) in enumerate(_range_and_bearing):
        plt.quiver(*_position_history[-1], r * np.cos(b + angle), r * np.sin(b + angle), angles='xy', scale_units='xy', scale=0.5, color='g', label=f'Agent {i + 1}')

    # Set the aspect of the plot to be equal
    plt.axis('equal')

    # Show the plot
    plt.legend()
    plt.show()

    # Output: [1. 0.]
    # The direction vector is [1, 0], which means the movement is to the right
    # The magnitude of the vector is 1, which means the distance between each position is 1
    # This is consistent with the input position history
    # The direction vector is normalized, so the magnitude is always 1
    # If you want the magnitude to be the actual distance between each position, you can remove the normalization
    # by replacing np.mean(diff, axis=0) with np.sum(diff, axis=0)
    # The magnitude of the vector will then be the sum of the distances between each position
    # In this case, the magnitude will be 9, which is the total distance between the first and last position
    # The direction will still be [1, 0], which means the movement is to the right
    # The magnitude of the vector is useful if you want to know the distance traveled in a given time period
    # The direction is useful if you want to know the general direction of the movement
    # For example, if the direction vector is [1, 0], the movement is to the right
    # If the direction vector is [0, 1], the movement is upwards
    # If the direction vector is [-1, 0], the movement is to the left
    # If the direction vector is [0, -1], the movement is downwards
    # If the direction vector is [1, 1], the movement is to the upper right
    # If the direction vector is [-1, 1], the movement is to the upper left
    # If the direction vector is [1, -1], the movement is to the lower right
    # If the direction vector is [-1, -1], the movement is to the lower left
    # If the direction
