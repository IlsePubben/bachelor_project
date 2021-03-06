import sys
from snakeGame import util 
# from snakeGame import visual_game
from learning import algorithms, neuralNetwork
import parameters as param
import stats
from keras.models import load_model

if __name__ == '__main__': 
    util.handle_command_line_options(sys.argv[1:])
    param.show_parameters()
    if param.algorithm == 'q-learning':
        # visual_game.pyglet.clock.schedule_interval(algorithms.q_learning,frequency)
        # performed as often as possible 
        # visual_game.pyglet.clock.schedule(algorithms.q_learning)
        model = neuralNetwork.NeuralNetwork(100,4, param.learning_rate_q)
        while param.epoch < param.max_epochs:
            algorithms.q_learning(1,model)
        param.epoch = 0
        previousEpoch = 0
        while param.epoch <= 100:
            if param.epoch != previousEpoch:
                previousEpoch = param.epoch
            algorithms.test(model)
        # print(stats.test_points)
        util.save_model(model)

    elif param.algorithm == 'qv-learning':
        vmodel = neuralNetwork.NeuralNetwork(100,1, param.learning_rate_v)
        qmodel = neuralNetwork.NeuralNetwork(100,4, param.learning_rate_q)
        while param.epoch < param.max_epochs:
            algorithms.qv_learning(1,qmodel,vmodel)
        param.epoch = 0
        while param.epoch <= 100:
            algorithms.test(qmodel)
        util.save_model(qmodel)
    
    elif param.algorithm == 'qvmax-learning':
        vmodel = neuralNetwork.NeuralNetwork(100,1, param.learning_rate_v)
        qmodel = neuralNetwork.NeuralNetwork(100,4, param.learning_rate_q)
        while param.epoch < param.max_epochs:
            algorithms.qvmax_learning(1,qmodel,vmodel)
        util.save_model(qmodel)
        
    elif param.algorithm == "qva-learning": 
        qmodel = neuralNetwork.NeuralNetwork(100,4, param.learning_rate_q)
        vmodel = neuralNetwork.NeuralNetwork(100,1, param.learning_rate_v)
        amodel = neuralNetwork.NeuralNetwork(100,4, param.learning_rate_a)
        while param.epoch < param.max_epochs:
            algorithms.qva_learning(1,qmodel,vmodel,amodel)
        param.epoch = 0
        while param.epoch <= 100:
            algorithms.test(amodel)
        util.save_model(amodel)
    
    elif param.algorithm == "qvamax-learning": 
        qmodel = neuralNetwork.NeuralNetwork(100,4, param.learning_rate_q)
        vmodel = neuralNetwork.NeuralNetwork(100,1, param.learning_rate_v)
        amodel = neuralNetwork.NeuralNetwork(100,4, param.learning_rate_a)
        while param.epoch < param.max_epochs:
            algorithms.qvamax_learning(1,qmodel,vmodel,amodel)
        util.save_model(amodel)
    
    
    # elif param.algorithm == 'random':
    #     visual_game.pyglet.clock.schedule_interval(algorithms.random_actions,frequency)
    # elif param.algorithm == 'manual':
    #     visual_game.pyglet.clock.schedule_interval(algorithms.manual,frequency)
    #     visual_game.pyglet.app.run()
    # elif param.algorithm == "play":
    #     model = load_model(param.model_filepath)
    #     numGames = input("How many games do you want to run?\n")
    #     frequency = 1/10
    #     visual_game.pyglet.clock.schedule_interval(algorithms.play, frequency, model,int(numGames))
    #     visual_game.pyglet.app.run()
    #     print("Mean points: ", stats.mean(stats.last1000points))
    


