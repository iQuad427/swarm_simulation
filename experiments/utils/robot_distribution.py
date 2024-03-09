import numpy as np


def points_on_circle(n_dots: int, radius: float, center: (int, int) = (0, 0)) -> np.ndarray:
    """
    Generate a set of points on a circle.

    :param n_dots: Number of points to generate
    :param radius: The radius of the circle
    :param center: The center of the circle
    """

    # Generate n_dots random points (x, y)
    return np.array([
        (center[0] + radius * np.cos(2 * np.pi * i / n_dots),
         center[1] + radius * np.sin(2 * np.pi * i / n_dots))
        for i in range(n_dots)
    ])


def format_to_xml(pos):
    """
    Format a set of points to XML.

    <foot-bot id="{i}" rab_data_size="1000" >
        <body orientation="0,0,0" position="0,0,0" />
        <controller config="fdc" />
    </foot-bot>

    :param pos: The set of points
    """
    ids = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
           "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
           "U", "V", "W", "X", "Y", "Z"]

    xml = ""
    for i in range(pos.shape[0]):
        xml += f'<foot-bot id="fb{ids[i]}" rab_range="1000" rab_data_size="1000" >\n'
        xml += f'\t<body orientation="0,0,0" position="{pos[i, 0]},{pos[i, 1]},0" />\n'
        xml += '\t<controller config="fdc" />\n'
        xml += '</foot-bot>\n'

    return xml


if __name__ == '__main__':
    # Generate a set of 16 points
    points = points_on_circle(20, 1.8)

    print(format_to_xml(points))
