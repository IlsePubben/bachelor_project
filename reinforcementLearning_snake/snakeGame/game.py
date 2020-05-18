import numpy
import parameters as param
from snakeGame import util 

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
        new_state = numpy.zeros(2 * param.vision_size**2 + 2)
        # new_state = numpy.zeros(6)
        reward = param.reward_default
        
        new_head_location = (self.head_location[0] + action[0], self.head_location[1] + action[1])
        #out of bounds
        if util.out_of_bounds(new_head_location):
            # print("out of bounds")
            reward = param.reward_dead
            return new_state, reward
            
        self.head_location = new_head_location 
        
        #found apple
        if self.head_location == self.apple_location:
            reward = param.reward_apple
            self.spawn_apple()
            self.points += 1
            self.time_stuck = 0
        else: 
            del self.tail_locations[0]
        
        if self.on_tail(self.head_location):
            # print("ate self")
            reward = param.reward_dead
        
        self.update_state(new_state)
        # self.display_state(new_state)
        
        self.time_stuck += 1
        return new_state, reward
    
    def valid_location(self, location):
        # print(location)
        return (not util.out_of_bounds(location) and not self.on_tail(location))
    
    # #without vision grid 
    # def update_state(self, state):
    #     if not self.valid_location((self.head_location[0] - 1, self.head_location[1])):
    #         state[Snake_vision.UP] = 1
    #     if not self.valid_location((self.head_location[0], self.head_location[1] + 1)):
    #         state[Snake_vision.RIGHT] = 1
    #     if not self.valid_location((self.head_location[0] + 1, self.head_location[1])):
    #         state[Snake_vision.DOWN] = 1
    #     if not self.valid_location((self.head_location[0], self.head_location[1] - 1)):
    #         state[Snake_vision.LEFT] = 1
    #     #direction apple scaled between -2 and 2
    #     state[Snake_vision.APPLEDX] = (self.apple_location[1] - self.head_location[1]) / param.game_size * 2
    #     state[Snake_vision.APPLEDY] = (self.apple_location[0] - self.head_location[0]) / param.game_size * 2
    #     # if self.apple_location[0] > self.head_location[0]:
    #     #     state[Snake_vision.NORTH] = 1
    #     # elif self.apple_location[0] < self.head_location[0]:
    #     #     state[Snake_vision.SOUTH] = 1
    #     # if self.apple_location[1] > self.head_location[1]:
    #     #     state[Snake_vision.EAST] = 1
    #     # elif self.apple_location[1] < self.head_location[1]:
    #     #     state[Snake_vision.WEST] = 1
    
    #with vision grid
    def update_state(self, state):
        idx = 0
        offset = param.vision_size // 2
        for i in range(-offset, offset + 1):
            for j in range(-offset, offset + 1):
                location = (self.head_location[0] + i, self.head_location[1] + j)
                if (not self.valid_location(location)
                    and not (i==0 and j==0)):
                    state[idx] = 1
                elif (location == self.apple_location):
                    state[idx+param.vision_size**2] = 1
                idx += 1
    

        state[len(state) - 1] = (self.apple_location[1] - self.head_location[1]) / param.game_size * 2
        state[len(state) - 2] = (self.apple_location[0] - self.head_location[0]) / param.game_size * 2               
        
    
    def get_state(self):
        # state = numpy.zeros(8)
        # state = numpy.zeros(6)
        state = numpy.zeros(2*param.vision_size**2+2)
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

    # def place_tail(self, state):
    #     if len(self.tail_locations) > 0: 
    #         for location in self.tail_locations:
    #             if not util.out_of_bounds(location):
    #                 state[location] = snake_tail
    
    def display_state(self, state):
        obstacles = numpy.zeros((param.vision_size, param.vision_size))
        apple = numpy.zeros((param.vision_size, param.vision_size))
        for row in range(0,param.vision_size):
            for col in range(0,param.vision_size): 
                obstacles[row][col] = state[row*param.vision_size + col]
                apple[row][col] = state[param.vision_size**2 + row*param.vision_size + col]
        print("obstacles\n",obstacles)
        print("apple\n",apple)
        print("distance to apple ", state[len(state) -1], state[len(state)-1])
    
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
                
        