import numpy 
import parameters as param
from game import util 

snake_head = numpy.array([1,0,0])
snake_tail = numpy.array([0,1,0])
apple = numpy.array([0,0,1])
empty = numpy.array([0,0,0])


def random_location():
    return numpy.random.randint(0,param.game_size), numpy.random.randint(0,param.game_size)

def find_location(state, obj):
    idx = numpy.where(obj == 1)
    for i in range(param.game_size): 
        for j in range(param.game_size): 
            if state[i,j][idx] == 1:
                return i,j
    return util.lost

def initialise_state():
    state = numpy.zeros((param.game_size,param.game_size,3))
    #place snake head
    head_location = random_location()
    state[head_location] = snake_head
    #place apple
    placing_apple = True
    while placing_apple: 
        apple_location = random_location()
        if apple_location != head_location:
            state[apple_location] = apple 
            placing_apple = False
    return state 

def make_move(state, action):
    apple_location = find_location(state, apple)
    head_location = find_location(state, snake_head)
    state = numpy.zeros((param.game_size,param.game_size,3))
    new_location = (head_location[0] + action[0], head_location[1] + action[1])
    if  all(i < param.game_size and i >= 0 for i in new_location):
        print(new_location)
        state[new_location] = snake_head
    state[apple_location][2] = 1
    return state

def display_state(state): 
    grid = numpy.zeros((param.game_size,param.game_size),dtype=str)
    head_location = find_location(state, snake_head)
    apple_location = find_location(state, apple)
    for i in range(param.game_size): 
        for j in range(param.game_size): 
            grid[i,j] = '-'
    grid[find_location(state, apple)] = 'A'
    if head_location == apple_location:
        grid[head_location] = 'F'
    elif head_location != util.lost:
        grid[head_location] = 'H'
    
    
    print(grid)
            
    