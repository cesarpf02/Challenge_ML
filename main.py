import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm  # to use a progress bar
import matplotlib.pyplot as plt
from model import net
from utils import util_model
from utils import util_dicom
from utils import util_data
from data import data_prep

import torch
import torch.optim as optim
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Parameters
DATA_DIR = "./data/processed"
REPORT_DIR = "./reports/"

# Network Parameters
USE_DROPOUT=False
USE_BN=False
LR = 0.001
BATCH_SIZE = 2
EPOCHS = 5

# Data parameters
VAL_PERC = 0.3
IMG_SIZE=50
CHANNEL_SIZE =1

MODEL_NAME=f'channel_{CHANNEL_SIZE}-net-loss_bincross-lr_{LR}-bs_{BATCH_SIZE}-drop_{USE_DROPOUT}-bn_{USE_BN}'

# Upload data
interim_dir = './data/interim'
data_dir = './data/raw'
exp_train_name = 'EXP1_blind'
exp_test_name = 'EXP2_open'
labels_train_name ='labels_exp1_bin.csv'
labels_test_name ='labels_exp2_bin.csv'
train_labels = pd.read_csv(os.path.join(interim_dir, labels_train_name), sep=';')
test_labels = pd.read_csv(os.path.join(interim_dir, labels_test_name), sep=';')
file_train_names = os.listdir(os.path.join(data_dir, exp_train_name))
if '.DS_Store' in file_train_names:
    file_train_names.remove('.DS_Store')
file_test_names = os.listdir(os.path.join(data_dir, exp_test_name))
if '.DS_Store' in file_test_names:
    file_test_names.remove('.DS_Store')
SAMPLES_SIZE = 5 # Must be <100
PERC_VAL=0.2
TRAIN_SIZE=int(SAMPLES_SIZE*(1-PERC_VAL))
TEST_SIZE=int(SAMPLES_SIZE*PERC_VAL)
train_samples_net, train_labels_net = data_prep.create_samples(data_dir, exp_train_name, file_train_names, train_labels, TRAIN_SIZE)
test_samples_net, test_labels_net = data_prep.create_samples(data_dir, exp_test_name, file_test_names, test_labels, TEST_SIZE)


# We want to iterate over our data (in batches)
# Along with separating out our data, we also need to shape this data (view it, according to Pytorch) in the way Pytorch
# expects us (-1, IMG_SIZE, IMG_SIZE)


train_samples_net=np.array(train_samples_net)
train_labels_net=np.array(train_labels_net)
train_images_tensor = torch.Tensor(train_samples_net)
train_images_tensor=train_images_tensor/255.0
train_y=torch.Tensor(train_labels_net)
train_y = torch.LongTensor(train_y.long())
train_X=train_images_tensor.view(-1,1,64,4096)
print('trainX',train_X.shape)
print('trainY',train_y.shape)

test_samples_net=np.array(test_samples_net)
test_labels_net=np.array(test_labels_net)
test_images_tensor = torch.Tensor(test_samples_net)
test_images_tensor = test_images_tensor/255.0
test_y = torch.Tensor(test_labels_net)
test_y = torch.LongTensor(test_y.long())
test_X=test_images_tensor.view(-1,1,64,4096)
print('testX',test_X.shape)
print('testY',test_y.shape)
# Define the model
net = net.Net(use_dropout=USE_DROPOUT, use_bn=USE_BN, channel_input=CHANNEL_SIZE)
net.to(device)
print(net)

# Define the loss and optimizer
optimizer = optim.Adam(net.parameters(), lr=LR) # all the net parameters are controlled by the defined optimiser
loss_function = nn.CrossEntropyLoss() # nn.MSELoss()  # mean squared error

# Train the model
history = {
    'train_loss': [],
    'val_loss': []
}
for epoch in range(EPOCHS):
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()  # Set model to training mode
            data_len = len(train_X)
        else:
            net.eval()  # Set model to evaluate mode
            data_len = len(test_X)

        # Reinitalize loss per each epoch.
        running_loss = 0.0
        for i in tqdm(range(0, data_len, BATCH_SIZE)):  # from 0, to the len of x, stepping BATCH_SIZE at a time.
            if phase == 'train':
                batch_X = train_X[i : i + BATCH_SIZE].view(-1, 1, 64, 4096)
                print('batchX', batch_X.shape)
                batch_y = train_y[i : i + BATCH_SIZE]
                print('batchY', batch_y.shape)

            elif phase == 'val':
                batch_X = test_X[i: i + BATCH_SIZE].view(-1, 1, 64, 4096)
                print('batchX val', batch_X.shape)
                batch_y = test_y[i: i + BATCH_SIZE]
                print('batchY val', batch_y.shape)

            else:
                raise NotImplementedError

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad() # zero the parameter gradients

            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(batch_X)  # compute the output for a batch of data - forward
                print(outputs.shape)
                print(batch_y.shape)
                loss = loss_function(outputs, batch_y)  # Evaluate loss

            if phase == 'train': # backward + optimize only if in training phase
                loss.backward()  # Backward pass (Computes the gradient for each tensor)
                optimizer.step() # Update the weights

            # Statistics
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / data_len
        print(f"\nEpoch: {epoch}. Loss_{phase}: {epoch_loss}")

        # Update history
        if phase == 'train':
            history['train_loss'].append(epoch_loss)
        else:
            history['val_loss'].append(epoch_loss)

util_model.plot_loss(history, MODEL_NAME, REPORT_DIR)

# Evaluate the model
correct = 0
total = 0
net.eval()  # import if there are dropout layers in the model (that are set to OFF in testing phase)
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        inputs = test_X[i].view(-1, CHANNEL_SIZE, 64, 4096)
        net_out = net(inputs)[0]  # returns a list
        predicted_class = torch.argmax(net_out)
        
        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy: ", round(correct/total, 3))