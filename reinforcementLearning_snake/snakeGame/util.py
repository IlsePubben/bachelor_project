import numpy
import parameters as param


# class Actions(enum.Enum):
#     UP = [-1,0]
#     DOWN = [1,0]
#     LEFT = [0,-1]
#     RIGHT = [0,1]
actions = [ [-1,0],[1,0],[0,-1],[0,1] ]

def random_location():
    return numpy.random.randint(0,param.game_size), numpy.random.randint(0,param.game_size)

def out_of_bounds(location):
    return (location[0] < 0 or location[0] >= param.game_size or
            location[1] < 0 or location[1] >= param.game_size)
    