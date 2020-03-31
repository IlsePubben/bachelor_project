import sys, getopt 
from game import main_game

def usage():
    print("OPTIONS: \n -h --help\n -a --algorithm: random | manual ")

def handle_command_line_options(argv):
    try: 
        options, args = getopt.getopt(argv, "ha:", ["help", "algorithm="])
    except getopt.GetoptError as error:
        print(error)
        usage()
        sys.exit(2)
    algorithm = ''
    for option, value in options:
        if option in ("-h", "--help"):
            usage()
            sys.exit(2)
        elif option in ("-a", "--algorithm"):
            if value in ("random", "manual"):
                algorithm = value
            else: 
                usage()
                sys.exit(2)
    return algorithm 

if __name__ == '__main__': 
    algorithm = handle_command_line_options(sys.argv[1:])
    main_game.pyglet.clock.schedule_interval(main_game.update,1/8)
    epochs = 1
    main_game.pyglet.app.run()
