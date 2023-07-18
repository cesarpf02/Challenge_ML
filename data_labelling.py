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

def type_bin(row):
    if row["type"] in ["TB", "TM"]:
        return 0
    elif row["type"] in ["FB", "FM"]:
        return 1

interim_dir = './data/interim'
data_dir = './data/raw'

df = pd.read_csv(os.path.join(interim_dir, 'labels_exp1.csv'))
df["type-bin"] = df.apply(type_bin, axis=1)
df.to_excel(os.path.join(interim_dir, 'labels_exp1_bin.xlsx'), engine='openpyxl', index=False)

df = pd.read_csv(os.path.join(interim_dir, 'labels_exp2.csv'))
df["type-bin"] = df.apply(type_bin, axis=1)
df.to_excel(os.path.join(interim_dir, 'labels_exp2_bin.xlsx'), engine='openpyxl', index=False)
