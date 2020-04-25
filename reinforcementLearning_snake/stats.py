import matplotlib.pyplot as plt 
from statistics import mean 

average_points = list() 
last1000points = list()
max_points = 0
counter = 0

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
        print("Epoch: ", epoch, " average: ", average, "high score: ", max_points)
        max_points = 0
        
def plot_multiple_models(filepaths_label, epochs, epsilon0):
    x_axis = [i*100 for i in range(1,epochs//100)]
    for key in filepaths_label: 
        with open(key, "r") as file: 
            model = eval(file.readline())
        plt.plot(x_axis,model, label=filepaths_label[key] )
  
    plt.xlabel("Epoch")
    plt.ylabel("Points (average over 1000 epochs measured every 100 epochs)")
    plt.axvline(epsilon0, 0, 16, label='epsilon=0', c="BLACK")
    plt.legend()
    plt.show()