import sys, getopt 
import numpy as np
import parameters as param
import stats

actions = [ [-1,0],[1,0],[0,-1],[0,1] ]

def random_location():
    return np.random.randint(0,param.game_size), np.random.randint(0,param.game_size)

def out_of_bounds(location):
    return (location[0] < 0 or location[0] >= param.game_size or
            location[1] < 0 or location[1] >= param.game_size)

def epsilon_greedy_action_selection(values):
    if np.random.random() < param.epsilon:
        return np.random.randint(0,4) #choose random action 
    else: 
        return np.argmax(values) #choose best action 

def boltzmann_exploration(values): 
    val = values.flatten()
    exps = [np.exp(i/param.temperature) for i in val]
    probabilities = [i/sum(exps) for i in exps]
    action = np.random.choice(4, 1, p=probabilities)
    return action[0]

#a scheduler needs to accept these parameters but we're not using them
def annealing_learningrate(e,lr): 
    # print("learning rate:", param.lr_annealing_factor**param.epoch * param.lr_start, e )
    return param.lr_annealing_factor**param.epoch * param.lr_start

def save_model(model):
    filepath = ("outputs/" + param.algorithm + str(param.max_epochs) + 
                "-v" + str(param.vision_size) +
                "-e" + str(param.start_epsilon) + "-y" + str(param.discount_factor) + 
                "-lr" + str(param.lr_start) + "-lr" + str(param.lr_end))
    # model.mlp.save(filepath)
    print("model saved as ", filepath)
    filepath += ".txt"
    with open(filepath,"a") as file: 
        file.write(str(stats.average_points))
        file.write("\n")
    filepath += "_last"
    with open(filepath,"w") as file: 
        file.write(str(stats.average_points))
        file.write("\n")
    savefile = "outputs/first_q_values" + str(param.algorithm) +  str(param.vision_size)
    with open(savefile, "w") as file: 
        file.write(str(stats.first_q_value))
    savefile = "outputs/reward"  + str(param.algorithm) +  str(param.vision_size)
    with open(savefile, "w") as file: 
        file.write(str(stats.cumulative_rewards)) 
        
def usage():
    print("OPTIONS: \n -h --help\n -a --algorithm: random | manual | q-learning",
           "| qv-learning | qvmax-learning | qva-learning | qvamax-learning")
    print(" -e --epsilon: value between 0-1")
    print(" -y --discountFactor: value between 0-1")
    print(" -v --visionSize: size of vision grid")
    print(" -t --test: path to model")
    # print(" --lrQ: Learning rate Q_model. default=0.001")
    # print(" --lrV: Learning rate V_model. default=0.001")
    # print(" --lrA: Learning rate A_model. default=0.001")

def handle_command_line_options(argv):
    try: 
        options, args = getopt.getopt(argv, "ha:t:e:y:v:", ["help", "algorithm=", "test=", "epsilon=",
                                                          "discountFactor=", "visionSize=", "lrQ=", "lrV=", "lrA="])
    except getopt.GetoptError as error:
        print(error)
        usage()
        sys.exit(2)
    for option, value in options:
        if option in ("-h", "--help"):
            usage()
            sys.exit(2)
        elif option in ("-a", "--algorithm"):
            if value in ("random", "manual", "q-learning", "qv-learning", "qva-learning",
                         "qvmax-learning", "qvamax-learning"):
                param.algorithm = value 
            else: 
                usage()
                print("non-existing algorithm")
                sys.exit(2)
        elif option in ("-t", "--test"):
            param.algorithm = value
        elif option in ("-e", "--epsilon"): 
            param.start_epsilon = float(value)
            param.epsilon = param.start_epsilon
        elif option in ("-y", "--discountFactor"):
            param.discount_factor = float(value)
        elif option in ("-v", "--visionSize"):
            param.vision_size = int(value) 
        elif option == "--lrQ":
            param.learning_rate_q = float(value)
        elif option == "--lrV":
            param.learning_rate_v = float(value)
        elif option == "--lrA":
            param.learning_rate_a = float(value)
        else:
            usage()
            sys.exit(2)

def show_parameters():
    print("Running program with the following parameters:")
    print("Algorithm:",param.algorithm, "\nEpsilon:",param.epsilon,
          "\nVisiongrid size:",param.vision_size,
          "\nDiscount-factor:",param.discount_factor, 
          "\nLearning rate start:", param.lr_start,
          "\nLearning rate end:", param.lr_end)
