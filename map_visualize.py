import numpy as np
import matplotlib.pyplot as plt

path = './maps_6_caves/map_1_r15.txt'
data = np.loadtxt(path)

im = plt.imshow(data, cmap='coolwarm', vmin=np.min(data), vmax=np.max(data)*1.25)
plt.title('Map data from: ' + path)
plt.xlabel('x-axis (2m per pixel)')
plt.ylabel('y-axis (2m per pixel)')
plt.colorbar(im)
plt.show()

