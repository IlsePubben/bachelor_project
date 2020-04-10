import random
import numpy as np 
from learning import MLP
from snakeGame import util, visual_game, game
import parameters as param

epoch = 0
game_state = game.Game()
state = game_state.get_state()

counter = 0
average_points = 0
time_stuck = 0
max_points = 0

def q_learning(timestep):
    global epoch 
    global game_state
    global state 
    #initialize game
    if epoch == 0:
        epoch +=1
    
    #toggle for visualisation 
    # visual_game.visualize(game_state)
    # print(state)
    
    qValues = MLP.mlp.predict(state.reshape(1,8), batch_size=1)
    
    if np.random.random() < param.epsilon:
        action = np.random.randint(0,4) #choose random action 
    else: 
        action = np.argmax(qValues) #choose best action 
    
    new_state, reward = game_state.make_move(util.actions[action])
    new_qValues = MLP.mlp.predict(new_state.reshape(1,8), batch_size=1)
    maxQ = np.max(new_qValues)
    if reward == param.reward_dead: #terminal state
        update = reward
        on_death()
    else:
        update = reward + param.discount_factor * maxQ
    
    target_output = qValues 
    target_output[0][action] = update
    MLP.mlp.fit(state.reshape(1,8),target_output,batch_size=1,verbose=0)
    state = new_state
    
    if game_state.time_stuck > 3*param.game_size:
        print("got stuck")
        on_death()
    
    if param.epsilon > 0.01:
        param.epsilon -= (1/param.max_epochs)
    
    #toggle for visualisation 
    # if epoch >= param.max_epochs:
        # visual_game.end_visualization()
    
def random_actions(timestep):
    global epoch 
    global game_state
    #initialize game
    if epoch == 0:
        print("start game")
        epoch += 1
    if epoch >= param.max_epochs:
        visual_game.end_visualization()
        # game_state = game.Game()
    action = random.choice(util.actions)
    visual_game.visualize(game_state,action)
    game_state.display_game() 
    state, reward = game_state.make_move(action)
    if reward == param.reward_dead:
        epoch += 1
        visual_game.reset(epoch)
        game_state = game.Game()
    
def manual(timestep):
    global epoch 
    global game_state
    #initialize game
    if epoch == 0:
        print("start game")
        epoch += 1
    if epoch >= param.max_epochs:
        visual_game.end_visualization()
        # game_state = game.Game()
    key = input()
    if key == 'w':
        action = [-1,0]
    elif key == 's':
        action = [1,0]
    elif key == 'a':
        action = [0,-1]
    elif key == 'd':
        action = [0,1]
    else: 
        action = [0,0]
        print("did nothing")
    visual_game.visualize(game_state,action)
    # game_state.display_game() 
    state, reward = game_state.make_move(action)
    if reward == param.reward_dead:
        epoch += 1
        visual_game.reset(epoch)
        game_state = game.Game()

def test(timestep, model, numGames):
    global state
    global game_state
    global time_stuck
    qValues = model.predict(state.reshape(1,8),batch_size=1)
    action = np.argmax(qValues)
    state, reward = game_state.make_move(util.actions[action])
    
    if reward == param.reward_dead: 
        on_death()
        
    if epoch > numGames:
        visual_game.end_visualization()

    if game_state.time_stuck > 3*param.game_size:
        print("got stuck")
        on_death()
    
    if reward == param.reward_apple:
        time_stuck = 0
        game_state.spawn_apple()
        game_state.update_state(state)
    
    visual_game.visualize(game_state,action)

def on_death():
    global epoch 
    global average_points
    global counter
    global game_state
    global max_points
    average_points = (average_points * counter + game_state.points) / (counter + 1)
    counter += 1
    if counter == 100:
        print("Epoch: ", epoch, "average points: ", average_points, "max: ", max_points)
        average_points = 0
        max_points = 0
        counter = 0
    if game_state.points >= max_points:
        max_points = game_state.points 
    epoch += 1
    game_state = game.Game()
    visual_game.reset(epoch)