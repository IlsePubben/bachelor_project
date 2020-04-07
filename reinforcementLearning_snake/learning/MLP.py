from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation 
from keras.layers import LeakyReLU


mlp = Sequential()
#First hidden layer
mlp.add(Dense(500, input_dim=192, kernel_initializer='lecun_uniform'))
#mlp.add(Activation('relu'))
mlp.add(LeakyReLU(alpha=0.01))
#Second hidden layer
mlp.add(Dense(284,kernel_initializer='lecun_uniform'))
# mlp.add(Activation('relu'))
mlp.add(LeakyReLU(alpha=0.01))
#Output layer
mlp.add(Dense(4,kernel_initializer='lecun_uniform'))
mlp.add(Activation('linear'))

#Compilation 
rms = RMSprop(learning_rate=0.001)
mlp.compile(optimizer=rms, loss='mse')