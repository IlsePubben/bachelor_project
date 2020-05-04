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

def save_model(model):
    filepath = ("outputs/" + param.algorithm + str(param.max_epochs) + 
                "-e" + str(param.start_epsilon) + "-y" + str(param.discount_factor) + 
                "-lq" + str(param.learning_rate_q) + "-lv" + str(param.learning_rate_v) +
                "-la" + str(param.learning_rate_a))
    # model.mlp.save(filepath)
    print("model saved as ", filepath)
    filepath += ".txt"
    with open(filepath,"a") as file: 
        file.write(str(stats.average_points))
        file.write("\n")
        file.close()
        
def usage():
    print("OPTIONS: \n -h --help\n -a --algorithm: random | manual | q-learning",
           "| qv-learning | qvmax-learning | qva-learning | qvamax-learning")
    print(" -e --epsilon: value between 0-1")
    print(" -y --discountFactor: value between 0-1")
    print(" -t --test: path to model")
    print(" --lrQ: Learning rate Q_model. default=0.001")
    print(" --lrV: Learning rate V_model. default=0.001")
    print(" --lrA: Learning rate A_model. default=0.001")

def handle_command_line_options(argv):
    try: 
        options, args = getopt.getopt(argv, "ha:t:e:y:", ["help", "algorithm=", "test=", "epsilon=",
                                                          "discountFactor=", "lrQ=", "lrV=", "lrA="])
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
          "\nDiscount-factor:",param.discount_factor, 
          "\nLearning-rate_q:",param.learning_rate_q, "\t Learning_rate_v:",param.learning_rate_v,
          "\tLearning-rate-a:",param.learning_rate_a)
