import pyglet
from snakeGame import resources
#from snakeGame import game
import parameters as param

gridSize = 32

window = pyglet.window.Window(8*gridSize,8*gridSize)

snake_batch = pyglet.graphics.Batch()

# snake = snake.Snake((util.random_location(window.width), 
#                      util.random_location(window.height)), snake_batch)
                    
apple = pyglet.sprite.Sprite(img=resources.apple_image, x=0, y=0)

snakeHead = pyglet.sprite.Sprite(img=resources.head_image,x=0,y=0, batch=snake_batch)

snakeTail = []
for idx in range(param.game_size*param.game_size):
    tail = pyglet.sprite.Sprite(img=resources.snake_image,x=0,y=0,batch=snake_batch)
    tail.visible = False
    snakeTail.append(tail)

epoch_label = pyglet.text.Label("Epoch 1", x=window.width/2, y=window.height-32,
                                anchor_x='center', anchor_y='center')


epochs = 1
points = 0

@window.event
def on_draw():
    window.clear()
    snake_batch.draw()
    apple.draw()
    epoch_label.draw()
    
def visualize(game, action=[-1,0]):
    # rotate_head(action)
    snakeHead.x = gridSize/2 + game.head_location[1] * gridSize
    snakeHead.y = window.height - (gridSize/2 + game.head_location[0] * gridSize)
    
    apple.x = gridSize/2 + game.apple_location[1] * gridSize
    apple.y = window.height - (gridSize/2 + game.apple_location[0] * gridSize)
    idx = 0
    for tail in game.tail_locations:
        snakeTail[idx].x = gridSize/2 + tail[1] * gridSize
        snakeTail[idx].y = window.height - (gridSize/2 + tail[0] * gridSize)
        snakeTail[idx].visible = True
        idx += 1

def rotate_head(action):
    if action == [-1,0]:
        snakeHead.rotation = 0
    elif action == [1,0]:
        snakeHead.rotation = 180
    elif action == [0,-1]: 
        snakeHead.rotation = -90
    elif action == [0,1]:
        snakeHead.rotation = 90

def reset(epoch):
    for tail in snakeTail: 
        tail.visible = False
    epoch_label.text = "Epoch " + str(epoch)

def end_visualization():
    window.close()
    pyglet.app.exit()

# def update(timeStep):
#     action = select_action()
    
#     if  not is_alive(snake.head):
#         if epochs >= parameters.max_epochs:
#             window.close()
#             pyglet.app.exit()
#         else: 
#             reset()
    
#     snake.move(action.name)
    
#     if snake.found_apple(apple):
#         snake.eat_apple()
#         global points 
#         points += 1
#         spawn_apple()
    
# def select_action():
#     algorithm = "random"
#     if algorithm == "random":
#         action = actions.Actions(random.randint(1, 4))
#     elif algorithm == "manual":
#         action = manual_movement()
#     return action 

# def is_alive(obj):
#     #out of bounds 
#     minXY = util.gridSize/2
#     maxXY = window.width - util.gridSize/2
#     if obj.x < minXY or obj.x > maxXY or obj.y < minXY or obj.y > maxXY:
#         return False
#     return not snake.on_snake_tail(snake.head)
#     return True

# def spawn_apple():
#     valid_location = False
#     while (not valid_location):
#         apple.x = util.random_location(window.width)
#         apple.y = util.random_location(window.height)
#         valid_location = not snake.on_snake_tail(apple)

# def reset():
#     snake.tail.clear()
#     snake.head.x = util.random_location(window.width)
#     snake.head.y = util.random_location(window.height)
#     spawn_apple()
#     global epochs
#     global points 
#     points = 0
#     epochs += 1
#     epoch_label.text = "Epoch " + str(epochs)
    
# def manual_movement():
#     if key_handler[key.UP]: return actions.Actions(1)
#     if key_handler[key.DOWN]: return actions.Actions(2)
#     if key_handler[key.LEFT]: return actions.Actions(3)
#     if key_handler[key.RIGHT]: return actions.Actions(4)
#     return actions.Actions(5)

# if __name__ == '__main__': 
#     algorithm = util.handle_command_line_options(sys.argv[1:])
#     pyglet.clock.schedule_interval(update,1/8)
#     epochs = 1
#     pyglet.app.run()
