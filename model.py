import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        #self.fc3 = nn.Linear(hidden_size2, hidden_size1)
        self.fc4 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # First layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second layer with ReLU activation
        #x = self.dropout(x)  # Dropout layer
        #x = F.relu(self.fc3(x)) # Third layer with ReLU activation
        #x = self.dropout(x)  # Dropout layer
        x = self.fc4(x)  # Fourth layer, no activation function here
        return x

        
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
        
class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float)

        next_state = np.array(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float)

        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            # add batch dimension
            state = torch.unsqueeze(state, 0) # [1, 11]
            next_state = torch.unsqueeze(next_state, 0) # [1, 11]
            action = torch.unsqueeze(action, 0) # [1, 3]
            reward = torch.unsqueeze(reward, 0) # [1, ]
            done = (done, ) # tuple
            
        # 1: predicted Q values with current state
        pred = self.model(state)
        
        target = pred.clone() # clone the predicted Q values
        for idx in range(len(done)): # loop through the batch
            Q_new = reward[idx] # if done, that's all
            if not done[idx]: # if not done, add discounted future reward
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) # Q_new = r + y * max(next_predicted Q value)
                
            target[idx][torch.argmax(action[idx]).item()] = Q_new # target = pred.clone() -> target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        self.optimizer.zero_grad() # reset the optimizer
        loss = self.criterion(target, pred) # calculate the loss
        loss.backward() # backpropagation
        
        self.optimizer.step() # update the weights