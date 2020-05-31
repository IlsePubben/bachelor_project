import numpy as np 
from snakeGame import util, game
# from snakeGame import visual_game
import parameters as param
from keras.callbacks import LearningRateScheduler
import keras.backend as k 
import stats

game_state = game.Game()
state = game_state.get_state()

got_stuck = 0

step = 0 

# v_value = 0 #to speed up qv(a)-learning
# q_values = np.zeros((1,4)) #to speed up qvmax-learniing

q_callback = [LearningRateScheduler(util.annealing_learningrate)]
v_callback = [LearningRateScheduler(util.annealing_learningrate_v)]
a_callback = [LearningRateScheduler(util.annealing_learningrate_a)]

tmp = True
saved = False

cumulative_reward = 0

def q_learning(timestep, model): 
    global game_state
    global state 
    global got_stuck
    global cumulative_reward

    #toggle for visualisation 
    # visual_game.visualize(game_state)
    
    if (param.epoch == 0): 
        state = game_state.get_state()
        param.epoch += 1
    
    qValues = model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    action = util.epsilon_greedy_action_selection(qValues)
    # action = util.boltzmann_exploration(qValues)
    
    new_state, reward = game_state.make_move(util.actions[action])
    new_qValues = model.mlp.predict(new_state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    maxQ = np.max(new_qValues)
    cumulative_reward += reward
    if reward == param.reward_dead: #terminal state
        update = reward
        stats.first_q_value.append(qValues[0][0])
        stats.cumulative_rewards.append(cumulative_reward)
        cumulative_reward = 0
        stats.difference_q_values.append(np.amax(qValues) - np.amin(qValues))
        on_death()
    else:
        update = reward + param.discount_factor * maxQ
    
    # print(qValues[0][0], action)
    target_output = qValues 
    target_output[0][action] = update
    
    # print(target_output, "\n")
    
    global saved
    if param.epoch >= 17950 and param.epoch <= 18050: 
        # stats.target_outputs_v.append(update)
        stats.target_outputs_q.append((target_output.flatten()).tolist())
    if param.epoch == 18051 and saved==False: 
        filepath = "outputs/target_outputs_q" + str(param.vision_size)
        with open(filepath,"w") as file: 
            file.write(str(stats.target_outputs_q))
        # filepath = "outputs/target_outputs_v" + str(param.vision_size)
        # with open(filepath,"w") as file: 
        #     file.write(str(stats.target_outputs_v))
        saved = True
        print("target outputs saved")
    
    model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2),target_output,batch_size=1, epochs=1, callbacks=q_callback, verbose=0)
    state = new_state
    
    if game_state.time_stuck > 2000:
        got_stuck += 1
        stats.first_q_value.append(qValues[0][0])
        stats.cumulative_rewards.append(cumulative_reward)
        cumulative_reward = 0
        stats.difference_q_values.append(np.amax(qValues) - np.amin(qValues))
        on_death()
    
    #toggle for visualisation 
    # if param.epoch >= param.max_epochs:
        # visual_game.end_visualization()
    
def qv_learning(timestep, q_model, v_model):
    global state
    global game_state
    global got_stuck
    global cumulative_reward
    global step 
    
    if (param.epoch == 0): 
        state = game_state.get_state()
        param.epoch += 1
    
    q_values = q_model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    action = util.epsilon_greedy_action_selection(q_values)
    # action = util.boltzmann_exploration(q_values)
    
    new_state, reward = game_state.make_move(util.actions[action])
    new_vValue = v_model.mlp.predict(new_state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    cumulative_reward += reward
    if reward == param.reward_dead: #terminal state
        update = np.array([[reward]])
        stats.first_q_value.append(q_values[0][0])
        stats.cumulative_rewards.append(cumulative_reward)
        cumulative_reward = 0
        stats.difference_q_values.append(np.amax(q_values) - np.amin(q_values))
        on_death()
    else: 
        update = reward + param.discount_factor * new_vValue
        
    target_output = q_values
    target_output[0][action] = update[0][0]
    
    if step < 30: 
        step += 1
        stats.target_outputs_v.append(update[0][0])
        stats.target_outputs_q.append((target_output.flatten()).tolist())
    if step == 30: 
        step += 1
        filepath = "outputs/target_outputs_q" + str(param.vision_size)
        with open(filepath,"w") as file: 
            file.write(str(stats.target_outputs_q))
        filepath = "outputs/target_outputs_v" + str(param.vision_size)
        with open(filepath,"w") as file: 
            file.write(str(stats.target_outputs_v))
        print("target outputs saved")

    v_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), update, batch_size=1, epochs=1, callbacks=v_callback, verbose=0)
    q_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), target_output, batch_size=1, epochs=1, callbacks=q_callback, verbose=0)
    # print("lr_q:", k.eval(q_model.mlp.optimizer.lr), "lr_v:", k.eval(v_model.mlp.optimizer.lr))
    state = new_state
    
    if game_state.time_stuck > 2000:
        got_stuck += 1
        stats.first_q_value.append(q_values[0][0])
        stats.cumulative_rewards.append(cumulative_reward)
        cumulative_reward = 0
        stats.difference_q_values.append(np.amax(q_values) - np.amin(q_values))
        on_death()

def qva_learning(timestep, q_model, v_model, a_model):
    global state
    global game_state
    global got_stuck
    global cumulative_reward
    # global v_value
    
    if (param.epoch == 0): 
        state = game_state.get_state()
        param.epoch += 1
        
    v_value = v_model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    q_values = q_model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    a_values = a_model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    
    action = util.epsilon_greedy_action_selection(a_values)
    # action = util.boltzmann_exploration(a_values)
    
    new_state, reward = game_state.make_move(util.actions[action])
    new_vValue = v_model.mlp.predict(new_state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    cumulative_reward += reward
    if reward == param.reward_dead: #terminal state
        update = np.array([[reward]])
        stats.first_q_value.append(q_values[0][0])
        stats.cumulative_rewards.append(cumulative_reward)
        cumulative_reward = 0
        stats.difference_q_values.append(np.amax(q_values) - np.amin(q_values))
        on_death()
    else: 
        update = reward + param.discount_factor * new_vValue
    
    print(q_values)
    q_target = q_values - 0
    q_target[0][action] = update 
    a_target = q_values - v_value
    print (q_values, "\n")

        
    v_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), update, batch_size=1, epochs=1, callbacks=v_callback, verbose=0)
    q_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), q_target, batch_size=1, epochs=1, callbacks=q_callback, verbose=0)
    a_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), a_target, batch_size=1, epochs=1, callbacks=a_callback, verbose=0)
    
    state = new_state 
    # v_value = new_vValue
    
    if game_state.time_stuck > 2000:
        got_stuck += 1
        stats.first_q_value.append(q_values[0][0])
        stats.cumulative_rewards.append(cumulative_reward)
        cumulative_reward = 0
        stats.difference_q_values.append(np.amax(q_values) - np.amin(q_values))
        on_death()

# def qvmax_learning(timestep, q_model, v_model):
#     global state
#     global game_state
#     global got_stuck
#     global q_values
    
#     if param.epoch == 0:
#         q_values = q_model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
#     action = util.epsilon_greedy_action_selection(q_values)
    
#     new_state, reward = game_state.make_move(util.actions[action])
#     new_vValue = v_model.mlp.predict(new_state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
#     new_qValues = q_model.mlp.predict(new_state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
#     if reward == param.reward_dead: #terminal state
#         v_update = np.array([[reward]])
#         q_update = v_update
#         on_death()
#     else: 
#         v_update = np.array([[reward + param.discount_factor * np.max(new_qValues)]])
#         q_update = reward + param.discount_factor * new_vValue
    
#     target_output = q_values
#     target_output[0][action] = q_update 

#     v_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), v_update, batch_size=1, epochs=1, callbacks=callback, verbose=0)
#     q_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), target_output, batch_size=1, epochs=1, callbacks=callback, verbose=0)
#     state = new_state
#     q_values = new_qValues
    
    
#     if game_state.time_stuck > param.game_size**2:
#         got_stuck += 1
#         on_death()

# def qvamax_learning(timestep, q_model, v_model, a_model):
#     global state
#     global game_state
#     global got_stuck
#     global v_value
#     global q_values
    
#     if param.epoch == 0: 
#         v_value = v_model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
#         q_values = q_model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
#     a_values = a_model.mlp.predict(state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    
#     action = util.epsilon_greedy_action_selection(a_values)
#     new_state, reward = game_state.make_move(util.actions[action])
#     new_vValue = v_model.mlp.predict(new_state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
#     new_qValues = q_model.mlp.predict(new_state.reshape(1,2 * param.vision_size**2 + 2), batch_size=1)
    
#     if reward == param.reward_dead: #terminal state
#         q_update = np.array([[reward]])
#         v_update = q_update
#         on_death()
#     else: 
#         v_update = np.array([[reward + param.discount_factor * np.max(new_qValues)]])
#         q_update = reward + param.discount_factor * new_vValue
    
#     q_target = q_values
#     q_target[0][action] = q_update 
#     a_target = q_values - v_value
        
#     v_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), v_update, batch_size=1, epochs=1, callbacks=callback, verbose=0)
#     q_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), q_target, batch_size=1, epochs=1, callbacks=callback, verbose=0)
#     a_model.mlp.fit(state.reshape(1,2 * param.vision_size**2 + 2), a_target, batch_size=1, epochs=1, callbacks=callback, verbose=0)
    
#     state = new_state 
#     v_value = new_vValue
#     q_values = new_qValues
    
#     if game_state.time_stuck > param.game_size**2:
#         got_stuck += 1
#         on_death()
    


# def test(timestep, model, numGames):
#     global state
#     global game_state
#     global got_stuck
    
#     if (param.epoch == 0): 
#         state = game_state.get_state()
#         param.epoch += 1
    
#     qValues = model.predict(state.reshape(1,2 * param.vision_size**2 + 2),batch_size=1)
#     action = np.argmax(qValues)
#     state, reward = game_state.make_move(util.actions[action])
    
#     if reward == param.reward_dead: 
#         on_death()
        
#     if param.epoch > numGames:
#         visual_game.end_visualization()

#     if game_state.time_stuck > param.game_size**2:
#         got_stuck += 1
#         on_death()
    
#     if reward == param.reward_apple:
#         game_state.spawn_apple()
#         game_state.update_state(state)
    
#     visual_game.visualize(game_state,action)

def on_death(): 
    global game_state
    global got_stuck
    
    stats.add_points(param.epoch, game_state.points)
    
    if param.epoch % 100 == 0:
        # print("Got stuck ", got_stuck, " times")
        got_stuck = 0
    
    global tmp 
    if param.epsilon > param.final_epsilon:
        param.epsilon -= (param.start_epsilon/(param.max_epochs-2000))
    elif tmp: 
        print("NO LONGER DECREASING EPSILON: ", param.epoch, "e =", param.epsilon) 
        print("step for Q-values ", len(stats.target_outputs_q))
        tmp = False
    param.epoch += 1
    game_state = game.Game()
    # visual_game.reset(param.epoch)

# def random_actions(timestep): 
#     global game_state
#     #initialize game
#     if param.epoch == 0:
#         print("start game")
#         param.epoch += 1
#     if param.epoch >= param.max_epochs:
#         visual_game.end_visualization()
#         # game_state = game.Game()
#     action = random.choice(util.actions)
#     visual_game.visualize(game_state,action)
#     game_state.display_game() 
#     state, reward = game_state.make_move(action)
#     if reward == param.reward_dead:
#         param.epoch += 1
#         visual_game.reset(param.epoch)
#         game_state = game.Game()
    
# def manual(timestep): 
#     global game_state
#     #initialize game
#     if param.epoch == 0:
#         print("start game")
#         param.epoch += 1
#     if param.epoch >= param.max_epochs:
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
#         param.epoch += 1
#         visual_game.reset(param.epoch)
#         game_state = game.Game()