import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, use_dropout=True, use_bn=None, channel_input=1):
        super().__init__()  # run the init of parent class (nn.Module)

        self.use_dropout = use_dropout
        self.use_bn = use_bn

        # Convs layers
        self.conv1 = nn.Conv3d(channel_input, 32, kernel_size=5)  # input is channel_input image, 32 output channels/feature maps, 5x5x5 kernel / window
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5)  # input is 32 as the first layer outputs 32 channels/feature maps. At the current layer we apply 64 channels with 5x5x5 kernel-window
        self.conv3 = nn.Conv3d(64, 128, kernel_size=5)

        # Batch norm layers
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # We need to flatten the output from the last convolutional layer before we can pass it through a regular "dense"
        # layer (or what pytorch calls a linear layer)
        # Let's determine the actual shape of the flattened output after the convolutional layers.
        # We can just simply pass some fake data initially to just get the shape. We can then just use a flag basically to
        # determine whether to do this partial pass of data to grab the value. We could keep the code itself cleaner
        # by grabbing that value every time as well, but we'd rather have faster speeds and just do the calc one time
        x = torch.randn(channel_input, 64, 64, 64).view(-1, channel_input, 64, 64, 64)  # create random data
        self._to_linear = None
        self.convs(x)  # Useful to define how many neurons we have inputting the classification block of the net

        # Linear layer
        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.fc2 = nn.Linear(512, 2)  # 512 in, 2 out bc we're using 2 classes (True vs false).

        # Dropout layer
        self.drop = nn.Dropout(p=0.5)  # p is the dropout rate – the probability of a neuron being deactivated

    def convs(self, x):
        # The input is already normalized! Batch norm skipped the first layer!
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        if self._to_linear is None:  # if we have not yet calculated what it takes to flatten (self._to_linear), we want to do that.
            batch_size = x.shape[0]
            self._to_linear = x.view(batch_size, -1).shape[1]
        return x

    def forward(self, x):
        # Feature extraction.
        x = self.convs(x)

        # Reshape (.view == reshape, we flatten x before feeding the classification block).
        x = x.view(-1, self._to_linear)
        if self.use_dropout:
            x = self.drop(x)

        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.drop(x)

        x = self.fc2(x)  # this is output layer. No activation here.
        return F.softmax(x, dim=1)
