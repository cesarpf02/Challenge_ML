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
import torch
import torch.optim as optim
import torch.nn as nn
import random


# Preparation of the data for the learning
def data_prep(data_dir,exp_name, exp_labels, uuid):
    scan, spacing, orientation, origin, raw_slices = util_dicom.load_dicom(os.path.join(data_dir, exp_name, str(uuid)))
    # Find the regions of interest in our scan. Note that the (slice, x, y) fields in exp_labels define the center of the lesion.
    locations = exp_labels.loc[exp_labels['uuid']==uuid]
    # Here are the regions of interest -> the attack involves a cuboid of dimensione 64 x 64 x 64 with respect to the location
    cut_cubes = []
    label_cubes=[]
    for i in range(len(locations)):
        location = locations.iloc[i]
        coord = np.array([location['slice'], location['y'], location['x']])
        cut_cubes.append(util_data.cutCube(scan,coord,(64, 64, 64), spacing))
        loc = locations.iloc[i]
        label_cubes.append(loc['type-bin'])
    norm_cubes = []
    for j in range (len(cut_cubes)):
        norm_cubes.append(util_data.normalize_image(cut_cubes[j], min_value=-1000, max_value=2000))

    return norm_cubes, label_cubes

def create_samples(data_dir, exp_name, file_names, file_labels, DATA_SIZE):
    samples_csv = random.sample(file_names, DATA_SIZE)
    labels=[]
    samples=[]
    for sample in samples_csv:
        samples_net, labels_net= data_prep(data_dir, exp_name, file_labels, int(sample))
        for sampl in samples_net:
            samples.append(sampl)        
        for lab in labels_net:
            labels.append(int(lab))
    return samples, labels





