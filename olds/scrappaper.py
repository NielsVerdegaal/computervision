import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



data = "/home/n13lsv/Documents/MAV project"

img = Image.open("/home/n13lsv/Documents/MAV project/AE4317_2019_datasets/cyberzoo_aggressive_flight/20190121-144646/49081789.jpg" )
arr = np.array(img)

print(arr.shape)     # (hoogte, breedte, kanalen)
rotated = np.rot90(arr, k=1)



# Visualize
plt.figure()
plt.title("Mask")
plt.imshow(rotated, cmap="gray")
plt.axis("off")

plt.show()