import numpy
import parameters as param
from snakeGame import util 

snake_head = numpy.array([0.9,0.1,0.1])
snake_tail = numpy.array([0.1,0.9,0.1])
apple = numpy.array([0.1,0.1,0.9])
empty = numpy.array([0.1,0.1,0.1])

class Game():
    
    def __init__(self):
        self.tail_locations = list()
        self.head_location =( param.game_size//2, param.game_size//2 )
        self.initialize_tail()
        self.apple_location = self.head_location
        self.spawn_apple()
        self.time_stuck = 0
        self.points = 0
    
    def initialize_tail(self):
        for idx in range(0,param.initial_snake_length):
            self.tail_locations.append(self.head_location)
    
    def spawn_apple(self):
        while self.apple_location == self.head_location or self.on_tail(self.apple_location): 
            self.apple_location = util.random_location()
          
    def make_move(self, action):
        self.tail_locations.append(self.head_location)
        new_state = numpy.zeros((param.game_size,param.game_size,3))
        
        new_head_location = (self.head_location[0] + action[0], self.head_location[1] + action[1])
        #out of bounds
        if util.out_of_bounds(new_head_location):
            # print("out of bounds")
            reward = param.reward_dead
            return new_state, reward
        
        #a postive reward is received when we get closer to the apple
        if self.distance_to_apple(new_head_location) < self.distance_to_apple(self.head_location):
            reward = param.reward_approach_apple
        else:
            reward = param.reward_avoid_apple
            
        self.head_location = new_head_location 
        
        #found apple
        if self.head_location == self.apple_location:
            reward = param.reward_apple
            self.points += 1
            self.time_stuck = 0
        else: 
            del self.tail_locations[0]
        
        if self.on_tail(self.head_location):
            # print("ate self")
            reward = param.reward_dead
        
        self.update_state(new_state)
        
        self.time_stuck += 1
        return new_state, reward
    
    def update_state(self, state):
        # if self.head_location == self.apple_location:
        #     state[self.head_location] = snake_head + apple
        # else: 
        self.place_tail(state)
        state[self.head_location] += snake_head
        state[self.apple_location] += apple
        
    
    def get_state(self):
        state = numpy.zeros((param.game_size,param.game_size,3))
        self.update_state(state)
        return state
    
    def distance_to_apple(self,location):
        return ( abs(self.apple_location[0] - location[0]) +
                abs(self.apple_location[1] - location[1]) )
    
    def on_tail(self,location):
        for tail in self.tail_locations:
            if tail == location:
                return True
        return False

    def place_tail(self, state):
        if len(self.tail_locations) > 0: 
            for location in self.tail_locations:
                if not util.out_of_bounds(location):
                    state[location] = snake_tail
    
    def display_game(self): 
        grid = numpy.zeros((param.game_size,param.game_size),dtype=str)

        for i in range(param.game_size): 
            for j in range(param.game_size): 
                grid[i,j] = '-'
        grid[self.apple_location] = 'A'
        grid[self.head_location] = 'H'
        for tail in self.tail_locations:
            grid[tail] = 'T'
        print(grid)
                
        