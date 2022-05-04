from pathlib import Path
import os
from nnunet.utilities.file_conversions import convert_2d_segmentation_nifti_to_img

in_dir = '/home/fer/repos/nnUNet/save_directory/results/CE_Loss'
files = os.listdir(in_dir)

for f in files:

	if ".nii.gz" in f:
		print("Working on : ",f)
		path_nifti = os.path.join(in_dir,f)
		file_name = f.split(".")[0]
		out_dir = os.path.join(in_dir,file_name+".png")
		print("Saving as: ", out_dir)
		convert_2d_segmentation_nifti_to_img(path_nifti, out_dir)