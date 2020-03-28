import pyglet 
import resources

class Apple(pyglet.sprite.Sprite):
    
    def __init__(self, *args, **kwargs):
        super().__init__(img=resources.apple_image, *args, **kwargs)