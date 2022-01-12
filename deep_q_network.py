import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


"""
Dueling DQN composed of convolutional layers to process game-play frames. From there, 
we split the network into two separate streams, one for estimating the state-value(V(s)) and the other for
estimating state-dependent action advantages(A(s,a)).
"""


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        # Initialize the checkpoint properties: directory and file name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Constructing the blueprint of the convolutional layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # Extracting outputted convolutional layers dimensions
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        # Making two fully connected layers
        self.fc1 = nn.Linear(fc_input_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)
        # Defining the state-value variable
        self.V = nn.Linear(512, 1)
        # Defining the advantage action variable
        self.A = nn.Linear(512, n_actions)
        # Defining the optimizing RMSprop
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        # defining the loss function 'MSE'
        self.loss = nn.MSELoss()
        # selecting the device 'cuda' or 'cpu'
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # calculating the network (forward and backward) in the selected device
        self.to(self.device)

    # calculate the outputted convolutional layers dimensions
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A
        
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
