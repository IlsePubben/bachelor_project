reward_apple = 5
reward_dead = -10
reward_default = 0
reward_approach_apple = 0
reward_avoid_apple = 0
temperature = 1
initial_snake_length = 3
game_size = 12
vision_size = 5
max_epochs = 20000
epoch = 0

start_epsilon = 0.1
final_epsilon = 0
epsilon = start_epsilon
discount_factor = 0.99
lr_start = 0.005
lr_end = 0.0005 
lr_annealing_factor = (lr_end/lr_start)**(1/float(max_epochs))
learning_rate_q = lr_start
learning_rate_v = learning_rate_q
learning_rate_a = learning_rate_q


algorithm = "" 
