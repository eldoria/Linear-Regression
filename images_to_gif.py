### Taken from the tutorial https://ransakaravihara.medium.com/how-to-create-gifs-using-matplotlib-891989d0d5ea

import glob
import re

#read all the .png files in directory called `steps`
files = sorted(glob.glob(r"images/*.png"), key=lambda filename: int(re.search(r'\d+', filename).group()))

from PIL import Image
import numpy as np

image_array = []

for my_file in files:
    image = Image.open(my_file)
    image_array.append(image)

print('image_arrays shape:', np.array(image_array).shape)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the initial image
im = ax.imshow(image_array[0], animated=True)

def update(i):
    im.set_array(image_array[i])
    return im,

# Create the animation object
animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=100, blit=True,repeat_delay=10,)

# Show the animation
plt.show()

animation_fig.save("animated_GMM.gif")
