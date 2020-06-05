import sys, getopt, os, errno
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
    return param.lr_annealing_factor**param.epoch * param.lr_start
def annealing_learningrate_q(e,lr): 
    return annealing_learningrate(e,lr) * param.lrQ_modifier
def annealing_learningrate_v(e,lr): 
    # print("lr V:", annealing_learningrate(e,lr) / 3, e )
    return annealing_learningrate(e,lr) * param.lrV_modifier
def annealing_learningrate_a(e,lr): 
    # print("lr A:", annealing_learningrate(e,lr) * 3, e )
    return annealing_learningrate(e,lr) * param.lrA_modifier

def save_model(model):
    filepath = ("outputs/experiments/" + param.name + param.algorithm + str(param.max_epochs) + 
                "_v" + str(param.vision_size) +
                "_e" + str(param.start_epsilon) + "_y" + str(param.discount_factor) + 
                "_lrQ" + str(round(param.lr_start * param.lrQ_modifier, 5)) + "-" + str(round(param.lr_end * param.lrQ_modifier, 5)) +
                "_lrV" + str(round(param.lr_start * param.lrV_modifier, 5)) + "-" + str(round(param.lr_end * param.lrV_modifier, 5)) +
                "_lrA" + str(round(param.lr_start * param.lrA_modifier, 5)) + "-" + str(round(param.lr_end * param.lrA_modifier, 5)))
    
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    # model.mlp.save(filepath)
    
    filepath += ".txt"
    with open(filepath,"a") as file: 
        file.write(str(stats.average_points))
        file.write("\n")
    # filepath += "_last"
    # with open(filepath,"w") as file: 
    #     file.write(str(stats.average_points))
    #     file.write("\n")
    # savefile = "outputs/new/first_q_values_" + str(param.algorithm) +  str(param.vision_size)
    # with open(savefile, "w") as file: 
    #     file.write(str(stats.first_q_value))
    # savefile = "outputs/new/reward_"  + str(param.algorithm) +  str(param.vision_size)
    # with open(savefile, "w") as file: 
    #     file.write(str(stats.cumulative_rewards)) 
    # savefile = "outputs/new/Qdifference_"  + str(param.algorithm) +  str(param.vision_size)
    # with open(savefile, "w") as file: 
    #     file.write(str(stats.difference_q_values)) 
        
    print("model saved as ", filepath)
        
def usage():
    print("OPTIONS: \n -h --help\n -a --algorithm: random | manual | q-learning",
           "| qv-learning | qvmax-learning | qva-learning | qvamax-learning")
    print(" -e --epsilon: value between 0-1, used for epsilon-greedy exploration")
    print(" -y --discountFactor: value between 0-1")
    print(" -v --visionSize: size of vision grid")
    print(" -t --temperature: value between 0-1, used for boltzman exploration")
    print(" --lrBegin: Learning rate at the beginning of training")
    print(" --lrEnd: Learning rate at the end of training")
    print(" --lrQ: Factor used to modify lr_begin to get the learning rate for the Q model, default=1")
    print(" --lrV: Factor used to modify lr_begin to get the learning rate for the V model, default=1")
    print(" --lrA: Factor used to modify lr_begin to get the learning rate for the A model, default=1")
    print(" -n --name: A string that will be added to the output filename")

def handle_command_line_options(argv):
    try: 
        options, args = getopt.getopt(argv, "ha:t:e:y:v:n:", ["help", "algorithm=", "temperature=", "epsilon=",
                                                          "discountFactor=", "visionSize=", "lrBegin=", "lrEnd=", 
                                                          "lrQ=", "lrV=", "lrA=", "test=", "name="])
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
        elif option in ("-t", "--temperature"):
            param.temperature = float(value)
        elif option in ("-e", "--epsilon"): 
            param.start_epsilon = float(value)
            param.epsilon = param.start_epsilon
        elif option in ("-y", "--discountFactor"):
            param.discount_factor = float(value)
        elif option in ("-v", "--visionSize"):
            param.vision_size = int(value) 
        elif option == "--lrBegin":
            param.lr_start = float(value)
            # param.lr_annealing_factor = (param.lr_end/param.lr_start)**(1/float(param.max_epochs))
            # param.learning_rate_q = param.lr_start
            # param.learning_rate_v = param.lr_start
            # param.learning_rate_a = param.lr_start
        elif option == "--lrEnd":
            param.lr_end = float(value)
            # param.lr_annealing_factor = (param.lr_end/param.lr_start)**(1/float(param.max_epochs))
        elif option == "--lrQ":
            param.lrQ_modifier = float(value)
        elif option == "--lrV":
            param.lrV_modifier = float(value)
        elif option == "--lrA":
            param.lrA_modifier = float(value)
        elif option == "--test":
            param.algorithm = "test"
            param.model_filepath = value
        elif option in ("-n", "--name"):
            param.name = value 
        else:
            usage()
            sys.exit(2)
    param.set_depending_parameters()


    
