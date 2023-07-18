import sys;
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([
    "./",
    "./src/"
])
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import util_dicom
from utils import util_data

interim_dir = './data/interim'
data_dir = './data/raw'
exp_name = 'EXP1_blind'
labels_name='labels_exp1_bin.csv'
exp_labels = pd.read_csv(os.path.join(interim_dir, labels_name), sep=';')
uuid_to_upload = 8038 # uuid of each patient

scan, spacing, orientation, origin, raw_slices = util_dicom.load_dicom(os.path.join(data_dir, exp_name, str(uuid_to_upload)))
print('The CT scan has the dimension of', scan.shape,'  (z,y,x)')

img = scan[0]
img_norm = util_data.normalize_image(img, min_value=-1000, max_value=2000)

cv2.imshow('Normalized Image', img_norm.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find the regions of interest in our scan. Note that the (slice, x, y) fields in exp_labels define the center of the lesion.
locations = exp_labels.loc[exp_labels['uuid']==uuid_to_upload]

# Here are the regions of interest -> the attack involves a cuboid of dimensione 64 x 64 x 64 with respect to
# the location
cut_cubes = []
for i in range(len(locations)):
    location = locations.iloc[i]
    coord = np.array([location['slice'], location['y'], location['x']])
    cut_cubes.append(util_data.cutCube(scan,coord,(64, 64, 64)))

# Visualization
for cube in cut_cubes:
    plt.figure(num=None, figsize=(6, 6), dpi=200)
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(cube[i, :, :],cmap='bone')
    plt.show()