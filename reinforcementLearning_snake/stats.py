import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from statistics import mean 
import os
import numpy as np 
# from scipy.stats import shapiro, normaltest

average_points = list() 
last1000points = list()
test_points = list() 
max_points = 0
counter = 0
first_q_value = list() 
difference_q_values = list() 
cumulative_rewards = list()
target_outputs_q = list()
target_outputs_v = list() 

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
            
def plot_directory(directory, epochs, epsilon0, title=""):
    fig = plt.figure()
    x_axis = [i*100 for i in range(1,epochs//100)]
    labels = ["Q-learning", "QV-learning", "QVA-learning"]
    idx = 0
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            filepath = directory + filename
            average = average_list_file(filepath)
            std = get_standard_deviation(filepath)
        
            # plt.plot(x_axis, average, label=filename)
            plt.plot(x_axis, average, label=labels[idx])
            idx += 1
            plt.fill_between(x_axis, [a_i - s_i for a_i, s_i in zip(average,std)],[a_i + s_i for a_i, s_i in zip(average,std)], alpha=0.2)
            # with open(filepath, "r") as file:
            #     model = eval(file.readline())
            # plt.plot(x_axis, model, label=filename)
    plt.xlabel("Epoch")
    plt.ylabel("Points")
    plt.axvline(epsilon0, 0, 16, label='epsilon=0', c="BLACK")
    plt.xlim(-100, epochs+100)
    plt.legend(bbox_to_anchor=(0.0, 1), loc=2)
    plt.title(title)
    return fig
    # plt.savefig('line_plot.pdf') 
    # plt.show()
    
def plot_3x3(directory="", saveAs=""):
    path = directory + "/con_1:1:1/"
    one = plot_directory(path, 20000,18002,"lrQ=0.005 ; lrV=lrQ ; lrA=lrQ")
    path = directory + "/an_1:1:1/"
    two = plot_directory(path,20000,18002,"lrQ=0.005->0.0005 ; lrV=lrQ ; lrA=lrQ")
    path = directory + "/con_1:03:3/"
    three = plot_directory(path,20000,18002,"lrQ=0.005 ; lrV=lrQ/3 ; lrA=lrQ*3")
    path = directory + "/con_1:03:3/"
    four = plot_directory(path,20000,18002,"lrQ=0.005->0.0005 ; lrV=lrQ/3 ; lrA=lrQ*3")
    
    pdf = PdfPages(saveAs)
    pdf.savefig(one)
    pdf.savefig(two)
    pdf.savefig(three)
    pdf.savefig(four)
    pdf.close()
    
def plot_all_visionGrids(directory, title=""):
    x_axis = [i*100 for i in range(1,200)]
    fig, axs = plt.subplots(2,2, sharex='col', sharey='all')
    fig.tight_layout()
    
    v3_path = directory + "/3x3/"
    for filename in sorted(os.listdir(v3_path)):
        if filename.endswith(".txt"): 
            filepath = v3_path + filename
            average = average_list_file(filepath)
            std = get_standard_deviation(filepath)
            axs[0,0].plot(x_axis, average, label=filename)
            axs[0,0].fill_between(x_axis, [a_i - s_i for a_i, s_i in zip(average,std)],[a_i + s_i for a_i, s_i in zip(average,std)], alpha=0.2)
    v5_path = directory + "/5x5/"
    for filename in sorted(os.listdir(v5_path)):
        if filename.endswith(".txt"): 
            filepath = v5_path + filename
            average = average_list_file(filepath)
            std = get_standard_deviation(filepath)
            axs[1,0].plot(x_axis, average, label=filename)
            axs[1,0].fill_between(x_axis, [a_i - s_i for a_i, s_i in zip(average,std)],[a_i + s_i for a_i, s_i in zip(average,std)], alpha=0.2)
    v7_path = directory + "/7x7/"
    for filename in sorted(os.listdir(v7_path)):
        if filename.endswith(".txt"): 
            filepath = v7_path + filename
            average = average_list_file(filepath)
            std = get_standard_deviation(filepath)
            axs[1,1].plot(x_axis, average, label=filename)
            axs[1,1].fill_between(x_axis, [a_i - s_i for a_i, s_i in zip(average,std)],[a_i + s_i for a_i, s_i in zip(average,std)], alpha=0.2)
    axs[0,0].legend()
    axs[0,0].set_title("Vision grid = 3x3")
    axs[1,0].set_title("Vision grid = 5x5")
    axs[1,1].set_title("Vision grid = 7x7")
    axs[0,1].set_visible(False)
    
    for c in axs: 
        for ax in c: 
            ax.axvline(18000, 0, 16, label='epsilon=0', c="BLACK")
            ax.set_xlim(-100, 20100)
    axs[1,0].set_xlabel('Epoch')
    axs[1,1].set_xlabel('Epoch')
    axs[1,0].set_ylabel("Points (average over 1000 epochs measured every 100 epochs)")
    fig.suptitle(title)
    
    fig.show()

def plot_4_graphs(directory, title=""):
    x_axis = [i*100 for i in range(1,200)]
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
    fig.tight_layout()
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"): 
            filepath = directory + filename
            average = average_list_file(filepath)
            std = get_standard_deviation(filepath)
            axs[0,0].plot(x_axis, average, label=filename)
            axs[0,0].fill_between(x_axis, [a_i - s_i for a_i, s_i in zip(average,std)],[a_i + s_i for a_i, s_i in zip(average,std)], alpha=0.2)
    x_axis = [i for i in range(1,20000)]
    dir_rewards = directory + "/reward/"
    for filename in sorted(os.listdir(dir_rewards)):
        filepath = dir_rewards + filename
        with open(filepath, "r") as file: 
            y = eval(file.readline())
            axs[1,0].scatter(x_axis,y,label=filename,alpha=0.2)
    dir_firstQ = directory + "/firstQ/"
    for filename in sorted(os.listdir(dir_firstQ)):
        filepath = dir_firstQ + filename
        with open(filepath, "r") as file: 
            y = eval(file.readline())
            axs[0,1].scatter(x_axis,y,label=filename, alpha=0.2)
    dir_diff = directory + "/diff/"
    for filename in sorted(os.listdir(dir_diff)):
        filepath = dir_diff + filename
        with open(filepath, "r") as file: 
            y = eval(file.readline())
            axs[1,1].scatter(x_axis,y,label=filename, alpha=0.2)
    axs[0,0].legend()
    axs[0,0].set_title("Average points")
    axs[1,0].set_title("Rewards")
    axs[0,1].set_title("First Q-value")
    axs[1,1].set_title("Difference between Q-values")
    axs.set_xlabel('Epoch')
    axs.set_ylabel("Points (average over 1000 epochs measured every 100 epochs)")
    for c in axs: 
        for ax in c: 
            ax.axvline(18000, 0, 16, label='epsilon=0', c="BLACK")
    fig.suptitle(title)
    fig.show()

def table(q_targets, v_targets, title):
    fig = plt.figure(figsize=(8,5))
    fig.suptitle(title)
    ax1 = fig.add_subplot(121)
    with open(v_targets, "r") as file: 
        y = eval(file.readline())
        x = [i for i in range(0,30)]
    ax1.scatter(x,y)
    ax2 = fig.add_subplot(122)
    with open(q_targets, "r") as file: 
        y = eval(file.readline())
        # df = pd.DataFrame(y)
        font_size=11
        bbox=[0, 0, 1, 1]
        ax2.axis('off')
        mpl_table = ax2.table(cellText = y, rowLabels=[str(i) for i in range(0,30)], bbox=bbox, colLabels=[str(i) for i in range(0,4)])
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
    
    plt.show()  

def plot_q_values(filepath):
    x = [i for i in range(0,19631)]
    with open(filepath, "r") as file: 
        q_values = eval(file.readline())
        print(len(q_values))
        for i in range (0,4):
            action = [row[i] for row in q_values]
            label = "Action " + str(i)
            plt.scatter(x,action,label=label, alpha=0.2)
    plt.xlabel("Step")
    plt.ylabel("Q-target")
    plt.axvline(10199, 0, 16, label='epsilon=0', c="BLACK")
    plt.title("Q-values between epoch 17950-18050")
    plt.legend()
    plt.show()

def rewards(directory):
    x = [i for i in range(0,19999)]
    for filename in os.listdir(directory):
        filepath = directory + filename
        with open(filepath, "r") as file: 
            y = eval(file.readline())
            plt.scatter(x,y,label="q-learning rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

def first_q_values(directory):
    x = [i for i in range(0,19999)]
    for filename in os.listdir(directory):
        filepath = directory + filename
        with open(filepath, "r") as file: 
            y = eval(file.readline())
            print(len(y))
            plt.scatter(x,y,label="q-learning", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("First Q-value")
    plt.legend()
    plt.show()
    
################### Actual statistics ###################

# def statistical_info(directory): 
#     listOfFiles = list()
#     for (dirpath, dirnames, filenames) in os.walk(directory):
#         listOfFiles += [os.path.join(dirpath, file) for file in filenames]
#     for filepath in sorted(listOfFiles):
#         if filepath.endswith("testRun"):
#             data = []
#             with open(filepath, "r") as file: 
#                 lines = file.readlines()
#                 for line in lines:
#                     data.append(float(line))
#             print(filepath)
#             data = np.array(data)
#             print("Mean:", np.mean(data))
#             print("Standard deviation:", np.std(data))
#             # plt.boxplot(data)
#             # plt.show()
#             # normality test
#             normal = True
#             stat, p = shapiro(data)
#             print('Shapiro normality test: Statistics=%.3f, p=%.3f' % (stat, p))
#             # interpret
#             alpha = 0.05
#             if p <= alpha:
#             	normal = False
#             if len(data) >= 20:
#                 stat, p = normaltest(data)
#                 print('Agostino normality test: Statistics=%.3f, p=%.3f' % (stat, p))
#                 # interpret
#                 if p <= alpha:
#                 	normal = False
#             if normal: 
#                 print("Normally distributed")
#             else: 
#                 print("Not normal")
#             print('\n')