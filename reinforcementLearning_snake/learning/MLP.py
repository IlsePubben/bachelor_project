from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation 

mlp = Sequential()
#First hidden layer
mlp.add(Dense(376, input_dim=192, init='lecun_uniform'))
mlp.add(Activation('relu'))
#Second hidden layer
mlp.add(Dense(284,init='lecun_uniform'))
mlp.add(Activation('relu'))
#Output layer
mlp.add(Dense(4,init='lecun_uniform'))
mlp.add(Activation('linear'))

#Compilation 
rms = RMSprop(learning_rate=0.001)
mlp.compile(optimizer=rms, loss='mse')