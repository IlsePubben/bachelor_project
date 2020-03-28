import pyglet, random 
import snake, apple, util, resources, actions
from pyglet.window import key

window = pyglet.window.Window(8*util.gridSize,8*util.gridSize)

snake_batch = pyglet.graphics.Batch()

snake = snake.Snake((util.random_location(window.width), 
                     util.random_location(window.height)), snake_batch)
                    
apple = pyglet.sprite.Sprite(img=resources.apple_image, x=util.random_location(window.width),
                              y=util.random_location(window.height))


epoch_label = pyglet.text.Label("Epoch 1", x=window.width/2, y=window.height-32,
                                anchor_x='center', anchor_y='center')

key_handler = key.KeyStateHandler()
window.push_handlers(key_handler)

@window.event
def on_draw():
    window.clear()
    snake_batch.draw()
    apple.draw()
    epoch_label.draw()

def update(timeStep):
    action = actions.Actions(random.randint(1, 4))
    #action = manual_movement()
    
    if  not is_alive(snake.head):
        print("dead ",action.name, ' ', epochs, '\n')
        reset()
    
    snake.move(action.name)
    
    if snake.found_apple(apple):
        snake.eat_apple()
        spawn_apple()
    
  
def is_alive(obj):
    #out of bounds 
    minXY = util.gridSize/2
    maxXY = window.width - util.gridSize/2
    if obj.x < minXY or obj.x > maxXY or obj.y < minXY or obj.y > maxXY:
        print("out of bounds")
        return False
    return not on_snake_tail(snake.head)
    return True

def on_snake_tail(obj):
    for tail in snake.tail[:-1]:
        if tail.x == obj.x and tail.y == obj.y:
            return True
    return False

def spawn_apple():
    valid_location = False
    while (not valid_location):
        apple.x = util.random_location(window.width)
        apple.y = util.random_location(window.height)
        valid_location = not on_snake_tail(apple)

def reset():
    snake.tail.clear()
    snake.head.x = util.random_location(window.width)
    snake.head.y = util.random_location(window.height)
    spawn_apple()
    global epochs
    epochs += 1
    epoch_label.text = "Epoch " + str(epochs)
    
def manual_movement():
    if key_handler[key.UP]: return actions.Actions(1)
    if key_handler[key.DOWN]: return actions.Actions(2)
    if key_handler[key.LEFT]: return actions.Actions(3)
    if key_handler[key.RIGHT]: return actions.Actions(4)
    return actions.Actions(5)

if __name__ == '__main__': 
    pyglet.clock.schedule_interval(update,1/10)
    epochs = 1
    pyglet.app.run()
