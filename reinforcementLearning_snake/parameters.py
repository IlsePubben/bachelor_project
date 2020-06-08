reward_apple = 5
reward_dead = -10
reward_default = -0.01
reward_approach_apple = 0
reward_avoid_apple = 0
temperature = 0.1
initial_snake_length = 3
game_size = 12
vision_size = 5
max_epochs = 20
epoch = 0

start_epsilon = 0.05
final_epsilon = 0
epsilon = start_epsilon
discount_factor = 0.99
lr_start = 0.005
lr_end = 0.0005 
lr_annealing_factor = (lr_end/lr_start)**(1/float(max_epochs))
lrQ_modifier = 1
lrV_modifier = 1
lrA_modifier = 1
learning_rate_q = lr_start * lrQ_modifier
learning_rate_v = lr_start * lrV_modifier
learning_rate_a = lr_start * lrA_modifier

model_filepath = ""
algorithm = "" 
name = ""
saveDir = ""

def set_depending_parameters(): 
    global epsilon, lr_annealing_factor, learning_rate_q, learning_rate_v, learning_rate_a
    epsilon = start_epsilon
    lr_annealing_factor = (lr_end/lr_start)**(1/float(max_epochs))
    learning_rate_q = lr_start * lrQ_modifier
    learning_rate_v = lr_start * lrV_modifier
    learning_rate_a = lr_start * lrA_modifier
    
def show_parameters():
    print("Running program with the following parameters:")
    print("Algorithm:",algorithm, "\nEpsilon:",epsilon,
          "\nTemperature:",temperature,
          "\nVisiongrid size:",vision_size,
          "\nDiscount-factor:",discount_factor, 
          "\nLearning rate start:", lr_start,
          "\nLearning rate end:", lr_end, 
          "\nLrQ:", learning_rate_q,
          "\nLrV:", learning_rate_v,
          "\nLrA:", learning_rate_a,
          "\nName:", name)