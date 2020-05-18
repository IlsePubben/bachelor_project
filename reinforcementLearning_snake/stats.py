import matplotlib.pyplot as plt 
from statistics import mean 
import os
import numpy as np 

average_points = list() 
last1000points = list()
max_points = 0
counter = 0
first_q_value = list() 
cumulative_rewards = list()

def plot_average100():
    x_axis = [i*100 for i in range(1,len(average_points)+1)]
    plt.title("Average over last 1000 episodes, measured every 100 episodes")
    plt.ylabel("Points")
    plt.xlabel("Episodes")
    plt.plot(x_axis, average_points)
    plt.show()
    
def add_points(epoch, points):
    global max_points
    global counter
    last1000points.append(points)
    if len(last1000points) > 1000:
        del last1000points[0]
    
    if points > max_points:
        max_points = points
        # print("Epoch: ", epoch, "HIGH SCORE: ", max_points)
    
    counter += 1
    if counter == 100: 
        counter = 0
        average = mean(last1000points)
        average_points.append(average)
        # print("Epoch: ", epoch, " average: ", average, "high score: ", max_points)
        max_points = 0
 
def average_list_file(filepath): 
    lists = [[]]
    with open(filepath, "r") as file:
        while True:
            line = file.readline()
            if not line: 
                break 
            lists.append(eval(line))
    lists = lists[1:]
    lists = np.array(lists, dtype=float)
    average = np.average(lists,axis=0)
    return list(average)
    # filepath = filepath.replace(".txt", "")
    # filepath += "_averaged"
    # with open(filepath, "w") as file: 
    #     file.write(str(average))

def average_directory(directory):
    for filename in os.listdir(directory):
        filepath = directory + filename
        average_list_file(filepath)

def get_standard_deviation(filepath):
    lists = [[]]
    with open(filepath, "r") as file: 
        while True: 
            line = file.readline()
            if not line: 
                break
            lists.append(eval(line))
    lists = lists[1:]
    lists = np.array(lists, dtype=float)
    std = np.std(lists, axis=0)
    return list(std)
            
def plot_directory(directory, epochs, epsilon0):
    x_axis = [i*100 for i in range(1,epochs//100)]
    for filename in os.listdir(directory):
        filepath = directory + filename
        average = average_list_file(filepath)
        std = get_standard_deviation(filepath)
        # plt.errorbar(x_axis, average, yerr=std, capsize=2, label=filename)
        plt.plot(x_axis, average, label=filename)
        # with open(filepath, "r") as file:
        #     model = eval(file.readline())
        # plt.plot(x_axis, model, label=filename)
    plt.xlabel("Epoch")
    plt.ylabel("Points (average over 1000 epochs measured every 100 epochs)")
    plt.axvline(epsilon0, 0, 16, label='epsilon=0', c="BLACK")
    plt.legend()
    plt.show()

def rewards():
    x = [i for i in range(0,19718)]
    with open("q-learning_reward", "r") as file: 
        y = eval(file.readline())
        plt.scatter(x,y,label="q-learning rewards")
    x = [i for i in range(0,18987)]
    with open("qv-learning_reward", "r") as file: 
        y = eval(file.readline())
        plt.scatter(x,y,label="qv-learning rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

def first_q_values():
    x = [i for i in range(0,19718)]
    with open("q-learningfirst_q_values", "r") as file: 
        y = eval(file.readline())
        plt.scatter(x,y,label="q-learning", marker='o')
    print("one")
    x = [i for i in range(0,19718)]
    # with open("q-learning_reward", "r") as file: 
    #     y = eval(file.readline())
    #     plt.plot(x,y,label="q-learning rewards")
    print("two")
    x = [i for i in range(0,18987)]
    with open("qv-learningfirst_q_values", "r") as file: 
        y = eval(file.readline())
        plt.scatter(x,y,label="qv-learning")
    print("three")
    # with open("qv-learning_reward", "r") as file: 
    #     y = eval(file.readline())
    #     plt.plot(x,y,label="qv-learning rewards")
    print("four")
    plt.xlabel("Epoch")
    plt.ylabel("First Q-value")
    plt.legend()
    plt.show()