import numpy as np
from PIL import Image


directory_path = './wsi_patch.png'
img = Image.open(directory_path)
img_L = np.array(img.convert("L"))
color_0_0_0 = np.where(img_L == 247)[0].shape[0]
print(color_0_0_0)