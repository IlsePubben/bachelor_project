import random
import numpy as np 
from snakeGame import util, game
# from snakeGame import visual_game
import parameters as param
import stats

epoch = 1
game_state = game.Game()
state = game_state.get_state()

got_stuck = 0

v_value = 0 #to speed up qv(a)-learning
q_values = np.zeros((1,4)) #to speed up qvmax-learniing

tmp = True
def q_learning(timestep, model):
    global epoch 
    global game_state
    global state 
    global got_stuck

    #toggle for visualisation 
    # visual_game.visualize(game_state)
    # print(state)
    
    qValues = model.mlp.predict(state.reshape(1,52), batch_size=1)
    
    action = util.epsilon_greedy_action_selection(qValues)
    
    new_state, reward = game_state.make_move(util.actions[action])
    new_qValues = model.mlp.predict(new_state.reshape(1,52), batch_size=1)
    maxQ = np.max(new_qValues)
    if reward == param.reward_dead: #terminal state
        update = reward
        on_death()
    else:
        update = reward + param.discount_factor * maxQ
    
    target_output = qValues 
    target_output[0][action] = update
    model.mlp.fit(state.reshape(1,52),target_output,batch_size=1,verbose=0)
    state = new_state
    
    if game_state.time_stuck > param.game_size**2:
        got_stuck += 1
        on_death()
    
    #toggle for visualisation 
    # if epoch >= param.max_epochs:
        # visual_game.end_visualization()
    
def qv_learning(timestep, q_model, v_model):
    global state
    global epoch
    global game_state
    global got_stuck
    
    q_values = q_model.mlp.predict(state.reshape(1,52), batch_size=1)
    action = util.epsilon_greedy_action_selection(q_values)
    
    new_state, reward = game_state.make_move(util.actions[action])
    new_vValue = v_model.mlp.predict(new_state.reshape(1,52), batch_size=1)
    if reward == param.reward_dead: #terminal state
        update = np.array([[reward]])
        on_death()
    else: 
        update = reward + param.discount_factor * new_vValue
    
    target_output = q_values
    target_output[0][action] = update 

    v_model.mlp.fit(state.reshape(1,52), update, batch_size=1,verbose=0)
    q_model.mlp.fit(state.reshape(1,52), target_output, batch_size=1, verbose=0)
    state = new_state
    
    if game_state.time_stuck > param.game_size**2:
        got_stuck += 1
        on_death()

def qvmax_learning(timestep, q_model, v_model):
    global state
    global epoch
    global game_state
    global got_stuck
    global q_values
    
    if epoch == 0:
        q_values = q_model.mlp.predict(state.reshape(1,52), batch_size=1)
    action = util.epsilon_greedy_action_selection(q_values)
    
    new_state, reward = game_state.make_move(util.actions[action])
    new_vValue = v_model.mlp.predict(new_state.reshape(1,52), batch_size=1)
    new_qValues = q_model.mlp.predict(new_state.reshape(1,52), batch_size=1)
    if reward == param.reward_dead: #terminal state
        v_update = np.array([[reward]])
        q_update = v_update
        on_death()
    else: 
        v_update = np.array([[reward + param.discount_factor * np.max(new_qValues)]])
        q_update = reward + param.discount_factor * new_vValue
    
    target_output = q_values
    target_output[0][action] = q_update 

    v_model.mlp.fit(state.reshape(1,52), v_update, batch_size=1,verbose=0)
    q_model.mlp.fit(state.reshape(1,52), target_output, batch_size=1, verbose=0)
    state = new_state
    q_values = new_qValues
    
    
    if game_state.time_stuck > param.game_size**2:
        got_stuck += 1
        on_death()

def qvamax_learning(timestep, q_model, v_model, a_model):
    global state
    global epoch
    global game_state
    global got_stuck
    global v_value
    global q_values
    
    if epoch == 0: 
        v_value = v_model.mlp.predict(state.reshape(1,52), batch_size=1)
        q_values = q_model.mlp.predict(state.reshape(1,52), batch_size=1)
    a_values = a_model.mlp.predict(state.reshape(1,52), batch_size=1)
    
    action = util.epsilon_greedy_action_selection(a_values)
    new_state, reward = game_state.make_move(util.actions[action])
    new_vValue = v_model.mlp.predict(new_state.reshape(1,52), batch_size=1)
    new_qValues = q_model.mlp.predict(new_state.reshape(1,52), batch_size=1)
    
    if reward == param.reward_dead: #terminal state
        q_update = np.array([[reward]])
        v_update = q_update
        on_death()
    else: 
        v_update = np.array([[reward + param.discount_factor * np.max(new_qValues)]])
        q_update = reward + param.discount_factor * new_vValue
    
    q_target = q_values
    q_target[0][action] = q_update 
    a_target = q_values - v_value
        
    v_model.mlp.fit(state.reshape(1,52), v_update, batch_size=1,verbose=0)
    q_model.mlp.fit(state.reshape(1,52), q_target, batch_size=1, verbose=0)
    a_model.mlp.fit(state.reshape(1,52), a_target, batch_size=1, verbose=0)
    
    state = new_state 
    v_value = new_vValue
    q_values = new_qValues
    
    if game_state.time_stuck > param.game_size**2:
        got_stuck += 1
        on_death()
    
def qva_learning(timestep, q_model, v_model, a_model):
    global state
    global epoch
    global game_state
    global got_stuck
    global v_value
    
    if epoch == 0: 
        v_value = v_model.mlp.predict(state.reshape(1,52), batch_size=1)
    q_values = q_model.mlp.predict(state.reshape(1,52), batch_size=1)
    a_values = a_model.mlp.predict(state.reshape(1,52), batch_size=1)
    
    action = util.epsilon_greedy_action_selection(a_values)
    new_state, reward = game_state.make_move(util.actions[action])
    new_vValue = v_model.mlp.predict(new_state.reshape(1,52), batch_size=1)
    
    if reward == param.reward_dead: #terminal state
        update = np.array([[reward]])
        on_death()
    else: 
        update = reward + param.discount_factor * new_vValue
    
    q_target = q_values
    q_target[0][action] = update 
    a_target = q_values - v_value
        
    v_model.mlp.fit(state.reshape(1,52), update, batch_size=1,verbose=0)
    q_model.mlp.fit(state.reshape(1,52), q_target, batch_size=1, verbose=0)
    a_model.mlp.fit(state.reshape(1,52), a_target, batch_size=1, verbose=0)
    
    state = new_state 
    v_value = new_vValue
    
    if game_state.time_stuck > param.game_size**2:
        got_stuck += 1
        on_death()

# def test(timestep, model, numGames):
#     global state
#     global game_state
#     global got_stuck
#     qValues = model.predict(state.reshape(1,52),batch_size=1)
#     action = np.argmax(qValues)
#     state, reward = game_state.make_move(util.actions[action])
    
#     if reward == param.reward_dead: 
#         on_death()
        
#     if epoch > numGames:
#         visual_game.end_visualization()

#     if game_state.time_stuck > param.game_size**2:
#         got_stuck += 1
#         on_death()
    
#     if reward == param.reward_apple:
#         game_state.spawn_apple()
#         game_state.update_state(state)
    
#     visual_game.visualize(game_state,action)

def on_death():
    global epoch 
    global game_state
    global got_stuck
    
    stats.add_points(epoch, game_state.points)
    
    if epoch % 100 == 0:
        # print("Got stuck ", got_stuck, " times")
        got_stuck = 0
    
    global tmp 
    if param.epsilon > param.final_epsilon:
        param.epsilon -= (param.start_epsilon/(param.max_epochs-2000))
    elif tmp: 
        print("NO LONGER DECREASING EPSILON: ", epoch, "e =", param.epsilon) 
        tmp = False
    epoch += 1
    game_state = game.Game()
    # visual_game.reset(epoch)

# def random_actions(timestep):
#     global epoch 
#     global game_state
#     #initialize game
#     if epoch == 0:
#         print("start game")
#         epoch += 1
#     if epoch >= param.max_epochs:
#         visual_game.end_visualization()
#         # game_state = game.Game()
#     action = random.choice(util.actions)
#     visual_game.visualize(game_state,action)
#     game_state.display_game() 
#     state, reward = game_state.make_move(action)
#     if reward == param.reward_dead:
#         epoch += 1
#         visual_game.reset(epoch)
#         game_state = game.Game()
    
# def manual(timestep):
#     global epoch 
#     global game_state
#     #initialize game
#     if epoch == 0:
#         print("start game")
#         epoch += 1
#     if epoch >= param.max_epochs:
#         visual_game.end_visualization()
#         # game_state = game.Game()
#     key = input()
#     if key == 'w':
#         action = [-1,0]
#     elif key == 's':
#         action = [1,0]
#     elif key == 'a':
#         action = [0,-1]
#     elif key == 'd':
#         action = [0,1]
#     else: 
#         action = [0,0]
#         print("did nothing")
#     visual_game.visualize(game_state,action)
#     # game_state.display_game() 
#     state, reward = game_state.make_move(action)
#     if reward == param.reward_dead:
#         epoch += 1
#         visual_game.reset(epoch)
#         game_state = game.Game()