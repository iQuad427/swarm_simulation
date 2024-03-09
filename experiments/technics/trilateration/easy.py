from easy_trilateration.model import *
from easy_trilateration.least_squares import easy_least_squares
from easy_trilateration.graph import *

if __name__ == '__main__':
    arr = [Circle(100, 100, 50),
           Circle(100, 50, 50),
           # Circle(50, 50, 50),
           # Circle(50, 100, 50)
           ]

    result, meta = easy_least_squares(arr)

    print(result)
    # create_circle(result, target=False)
    draw(arr)
