from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
import numpy as np

height = 20
width = 20 
noise = np.zeros((width, height))
simplex = OpenSimplex(0)

for y in range(0, height):
    for x in range(0, width):
        noise[x][y] = int(10 * abs((simplex.noise2(x , y))))

print(noise)

plt.imshow(noise, cmap='hot', interpolation='nearest')
plt.show() 


