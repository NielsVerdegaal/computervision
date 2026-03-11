import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



data = "/home/n13lsv/Documents/MAV project"

img = Image.open("/home/n13lsv/Documents/MAV project/AE4317_2019_datasets/cyberzoo_aggressive_flight/20190121-144646/49081789.jpg" )
arr = np.array(img)

print(arr.shape)     # (hoogte, breedte, kanalen)
# rotated = np.rot90(arr, k=1)

R = arr[:, :, 0]
G = arr[:, :, 1]
B = arr[:, :, 2]

# Define dark green in RGB
mask = (G > 70) & (G < 180) & ((R) > 80) & ((G - B) > 80)

mask = mask.astype(bool)  # ensure boolean mask

# Create masked image using multiplication (no fancy indexing)
masked = img * mask[:, :, None]

# Visualize
plt.figure()
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.figure()
plt.title("Masked image")
plt.imshow(masked)
plt.axis("off")

plt.show()