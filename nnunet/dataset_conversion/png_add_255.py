import cv2
import numpy as np
import os

folder = '/home/fer/repos/nnUNet/save_directory/results/CE_Loss'

files = os.listdir(folder)

for f in files:
	if ".png" in f:
		img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED)
		img[img>0] = 255
		cv2.imwrite(os.path.join(folder, f), img)