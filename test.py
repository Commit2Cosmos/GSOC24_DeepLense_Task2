import numpy as np


img = np.random.rand(1,5,5)

# Broadcast the original array to the new shape
img = np.broadcast_to(img, (3,img.shape[1],img.shape[2]))

print(img)