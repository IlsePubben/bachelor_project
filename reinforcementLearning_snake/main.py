import sys, getopt 
from snakeGame import visual_game
from learning import algorithms, MLP
import parameters as param
from keras.models import load_model

def usage():
    print("OPTIONS: \n -h --help\n -a --algorithm: random | manual | q-learning")
    print(" -t --test: path to model")

def handle_command_line_options(argv):
    try: 
        options, args = getopt.getopt(argv, "ha:t:", ["help", "algorithm=", "test="])
    except getopt.GetoptError as error:
        print(error)
        usage()
        sys.exit(2)
    for option, value in options:
        if option in ("-h", "--help"):
            usage()
            sys.exit(2)
        elif option in ("-a", "--algorithm"):
            if value in ("random", "manual", "q-learning"):
                print("test")
                return value
            else: 
                usage()
                print("non-existing algorithm")
                sys.exit(2)
        elif option in ("-t", "--test"):
            return value
        else:
            usage()
            sys.exit(2)

if __name__ == '__main__': 
    algorithm = handle_command_line_options(sys.argv[1:])
    # print(algorithm)
    frequency = 1/7
    if algorithm == 'q-learning':
        # visual_game.pyglet.clock.schedule_interval(algorithms.q_learning,frequency)
        # performed as often as possible 
        # visual_game.pyglet.clock.schedule(algorithms.q_learning)
        while(algorithms.epoch < param.max_epochs):
            algorithms.q_learning(1)
    elif algorithm == 'random':
        visual_game.pyglet.clock.schedule_interval(algorithms.random_actions,frequency)
    elif algorithm == 'manual':
        visual_game.pyglet.clock.schedule_interval(algorithms.manual,frequency)
    else:
        # print(algorithm)
        model = load_model(algorithm)
        numGames = input("How many games do you want to run?\n")
        visual_game.pyglet.clock.schedule_interval(algorithms.test, frequency, model,int(numGames))
    # visual_game.pyglet.app.run()
    filepath = algorithm + str(param.max_epochs)
    MLP.mlp.save(filepath)
    print("model saved")
