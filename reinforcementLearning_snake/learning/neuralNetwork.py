from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation 
import parameters as param
# from keras.layers import LeakyReLU

class NeuralNetwork():
    
    def __init__(self, hidden_nodes, output_nodes, learning_rate):
        self.mlp = Sequential()
        #First hidden layer
        self.mlp.add(Dense(hidden_nodes, input_dim=2 * param.vision_size**2 + 2, kernel_initializer='lecun_uniform'))
        self.mlp.add(Activation('sigmoid'))
        # self.mlp.add(LeakyReLU(alpha=0.01))
        #Second hidden layer
        # mlp.add(Dense(28,kernel_initializer='lecun_uniform'))
        # mlp.add(Activation('relu'))
        # mlp.add(LeakyReLU(alpha=0.01))
        #Output layer
        self.mlp.add(Dense(output_nodes,kernel_initializer='lecun_uniform'))
        self.mlp.add(Activation('linear'))
        
        #Compilation 
        rms = RMSprop(learning_rate=learning_rate)
        self.mlp.compile(optimizer=rms, loss='mse')
    