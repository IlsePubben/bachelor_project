import pyglet 
from snakeGame import resources, util 

class Snake():
    
    def __init__(self, position=(0,0), batch = None):
        self.position_head = position 
        self.image = resources.snake_image
        self.batch = batch
        self.tail = [] 
        self.head = pyglet.sprite.Sprite(resources.head_image, x=position[0],
                                         y=position[1], batch=self.batch)
        self.velocity_x = 0
        self.velocity_y = 0
    
    def move(self, action):
        
        #used for manual movement only 
        if action == "IDLE":
            return 
        
        #move tail
        if len(self.tail) > 0:
            new_tail_position = (self.head.x, self.head.y)
        
        #move head
        if action == "UP": 
            self.head.y += util.gridSize
            self.head.rotation = 0
        elif action == "DOWN":
            self.head.y -= util.gridSize
            self.head.rotation = 180
        elif action == "RIGHT": 
            self.head.x += util.gridSize
            self.head.rotation = 90
        elif action == "LEFT":
            self.head.x -= util.gridSize
            self.head.rotation = -90
        
        #move tail 
        if len(self.tail) > 0: 
            new_tail = self.tail.pop(0)
            new_tail.x = new_tail_position[0]
            new_tail.y = new_tail_position[1]
            self.tail.append(new_tail)
            
    def found_apple(self, apple):
        return (self.head.x == apple.x and self.head.y == apple.y)
    
    
    def on_snake_tail(self,obj):
        for tail in self.tail[:-1]:
            if tail.x == obj.x and tail.y == obj.y:
                return True
        return False
    
    def eat_apple(self): 
        new_tail = pyglet.sprite.Sprite(img=resources.snake_image, x=self.head.x,
                                        y=self.head.y, batch=self.batch)
        self.tail.append(new_tail)
        