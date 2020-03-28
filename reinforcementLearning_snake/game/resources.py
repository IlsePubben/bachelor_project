import pyglet 

def center_image(image): 
    image.anchor_x = image.width/2
    image.anchor_y = image.height/2

pyglet.resource.path = ["resources"]
pyglet.resource.reindex()

apple_image = pyglet.resource.image("sprite_apple.png")
center_image(apple_image)
snake_image = pyglet.resource.image("sprite_body.png")
center_image(snake_image)
head_image = pyglet.resource.image("sprite_head.png")
center_image(head_image)


