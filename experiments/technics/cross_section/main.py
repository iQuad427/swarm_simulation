import random
from collections import defaultdict
from sklearn.manifold import MDS

import matplotlib.pyplot as plt


class Node:
    def __init__(self, name: str, real_position: tuple = None):
        self.real_position = real_position
        self.name = name
        self.matrix = defaultdict(dict)
        self.matrix[self][self] = 0
        self.last_update = defaultdict(int)
        self.speed = 5

    def attach_beacon(self, beacon: 'Node'):
        self.matrix[beacon] = {}
        self.matrix[beacon][self] = compute_distance(self, beacon)
        self.matrix[self][beacon] = compute_distance(self, beacon)
        self.last_update[beacon] = 0

    def update_matrix(self, emitter: 'Node', about: 'Node', distance: float):
        self.matrix[emitter][about] = distance
        self.matrix[about][emitter] = distance
        self.matrix[emitter][emitter] = 0
        self.matrix[about][about] = 0
        self.last_update[emitter] = 0

    def tick(self):
        for node in self.matrix:
            self.last_update[node] += 1

    def __repr__(self):
        return f"Node {self.name}"

    def build_2d_representation(self):
        mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, metric=True, random_state=42)
        print([
            [self.matrix[node_1].get(node_2) for node_2 in self.matrix]
            for node_1 in self.matrix
        ]
        )
        embedding = mds.fit_transform(
            [
                [self.matrix[node_1][node_2] for node_2 in self.matrix]
                for node_1 in self.matrix
            ]
        )

        positions_of_nodes = {node: embedding[i] for i, node in enumerate(self.matrix)}

        return positions_of_nodes

    def get_uncertainty(self, node: 'Node'):
        if node.name == 'Beacon':
            return 0
        return self.last_update[node] * self.speed


def compute_distance(node_1: Node, node_2: Node):
    return ((node_1.real_position[0] - node_2.real_position[0]) ** 2 + (
            node_1.real_position[1] - node_2.real_position[1]) ** 2) ** 0.5


def update_all_nodes(nodes: list):
    for k in range(len(nodes)):
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                distance = compute_distance(nodes[i], nodes[j])
                nodes[k].update_matrix(nodes[i], nodes[j], distance)


def node_a_gives_all_its_distances_to_node_b(node_a: Node, node_b: Node):
    for node in node_a.matrix:
        node_b.update_matrix(node_a, node, node_a.matrix[node_a][node])


def tick_all_nodes(nodes: list):
    for node in nodes:
        node.tick()


if __name__ == '__main__':
    _nodes = [
        Node(str(i), (random.randint(0, 100), random.randint(0, 100)))
        for i in range(10)
    ]

    # Add beacon node
    _nodes.append(Node('Beacon', (0, 0)))

    # Attach all nodes to the beacon
    for node in _nodes:
        node.attach_beacon(_nodes[-1])

    update_all_nodes(_nodes)

    tick_all_nodes(_nodes)
    tick_all_nodes(_nodes)

    node_a_gives_all_its_distances_to_node_b(_nodes[1], _nodes[0])

    tick_all_nodes(_nodes)

    main_node = _nodes[0]
    positions = main_node.build_2d_representation()

    ax = plt.gca()
    ax.cla()  # clear things for fresh plot

    # Axis aspect ratio should be equal
    ax.set_aspect('equal', adjustable='box')

    new_current_positions = []

    # Make the size of the fig (20, 20)
    fig = plt.gcf()
    fig.set_size_inches(100, 100)

    for node, position in positions.items():
        # Plot real position in blue
        ax.scatter(position[0], position[1])
        ax.text(position[0], position[1], node.name)
        # Plot uncertainty as red circle around each node
        uncertainty = main_node.get_uncertainty(node)
        ax.add_patch(plt.Circle(position, uncertainty, color='r', fill=False))
        # Plot the circle of which the center is the beacon, and the radius is the distance to the beacon
        beacon_position = positions[_nodes[-1]]
        ax.add_patch(plt.Circle(beacon_position, main_node.matrix[node][_nodes[-1]], color='g', fill=False))

        move_x = random.randint(-uncertainty // 2, uncertainty // 2)
        move_y = random.randint(-uncertainty // 2, uncertainty // 2)
        new_position = (position[0] + move_x, position[1] + move_y)
        new_current_positions.append(new_position)
        # Compute distance of move_x, move_y to beacon
        distance_to_beacon = ((position[0] + move_x - beacon_position[0]) ** 2 + (
                    position[1] + move_y - beacon_position[1]) ** 2) ** 0.5
        ax.scatter(position[0] + move_x, position[1] + move_y, color='r')
        ax.text(position[0] + move_x, position[1] + move_y, node.name + '\'')
        ax.add_patch(plt.Circle(beacon_position, distance_to_beacon, color='r', fill=False, linewidth=30, alpha=0.1))

    main_node_new_position = new_current_positions[0]

    for node, new_position in zip(positions, new_current_positions):
        # Skip if node is Beacon or main node
        if node.name == 'Beacon' or node.name == '0':
            continue

        # Plot circle centered around current new position, with radius the distance to the new position of the main node
        distance_to_main_node = ((new_position[0] - main_node_new_position[0]) ** 2 + (
                    new_position[1] - main_node_new_position[1]) ** 2) ** 0.5
        ax.add_patch(plt.Circle(new_position, distance_to_main_node, color='b', fill=False, linewidth=20,
                                alpha=0.1))

    plt.tight_layout()
    # Save the plot to a file
    plt.savefig('plot.png')