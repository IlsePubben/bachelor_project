import random 

gridSize = 32

def random_location(axis_length):
    return gridSize/2 + random.randint(gridSize, axis_length-gridSize) // gridSize * gridSize

