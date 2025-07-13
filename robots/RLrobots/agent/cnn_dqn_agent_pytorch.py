import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import random
from collections import deque
import pickle
import os
from PIL import Image
import base64
import io

class CNNNetwork(nn.Module):
    def __init__(self, image_channels=4, action_size=8, hidden_size=512):
        super(CNNNetwork, self).__init__()
        
        # CNN layers for 4-channel image processing using Sequential
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(image_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate the size after CNN layers
        # Assuming input is 128x128:
        # Conv1: (128 - 8) / 4 + 1 = 31 -> 31x31x32
        # Conv2: (31 - 4) / 2 + 1 = 14 -> 14x14x64  
        # Conv3: (14 - 3) / 1 + 1 = 12 -> 12x12x64
        # So final spatial size is 12x12 with 64 channels
        cnn_output_size = 64 * 12 * 12
        
        # Fully connected layers using Sequential
        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x_image):
        # Process 4-channel image through CNN layers
        x_image = self.cnn_layers(x_image)
        
        # Flatten spatial features
        x_image = x_image.view(x_image.size(0), -1)
        
        # Process through fully connected layers
        return self.fc_layers(x_image)

class CNNDQNAgent:
    def __init__(self, image_channels=4, action_size=8, learning_rate=0.0015, epsilon=1, 
                 epsilon_min=0.01, epsilon_decay=0.9, memory_size=20_000, device='auto'):
        """
        Initialize CNN DQN Agent for 4-channel image input
        
        Args:
            image_channels: Number of image channels (4 for frame buffer)
            action_size: Size of action space (8 for new RobotAction actions)
            learning_rate: Learning rate for optimizer
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            memory_size: Size of replay memory
            device: Device to run on ('cpu' or 'cuda' or 'auto')
        """
        self.image_channels = image_channels
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 64  # Reduced batch size for image processing
        self.gamma = 0.99  # Discount factor
        self.update_target_frequency = 100
        self.update_counter = 0
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = CNNNetwork(image_channels, action_size).to(self.device)
        self.target_network = CNNNetwork(image_channels, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        print(f"CNN DQN Agent initialized with {action_size} actions on {self.device}")
        print(f"4-channel image channels: {image_channels}")

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state_image, action, reward, next_state_image, done):
        """Store experience in replay memory"""
        self.memory.append((state_image, action, reward, next_state_image, done))
    
    def get_q_values(self, state_image):
        state_tensor = torch.FloatTensor(state_image).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values
    
    def get_optimal_action(self, state_image):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_image).unsqueeze(0).to(self.device)
            q_values = self.target_network(state_tensor)
            return q_values.argmax().item()
    
    def act(self, state_image):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state_image).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states_image = torch.FloatTensor([experience[0] for experience in batch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in batch]).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in batch]).to(self.device)
        next_states_image = torch.FloatTensor([experience[3] for experience in batch]).to(self.device)
        dones = torch.IntTensor([experience[4] for experience in batch]).to(self.device)
        
        # Current Q values
        q_values = self.q_network(states_image)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_image).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        self.epsilon = self.linear_schedule(self.epsilon, self.epsilon_min, 10000000*0.1, self.update_counter)
        
        return loss.item()
    
    def save_model(self, filename='cnn_dqn_model_pytorch.pth'):
        """Save model weights and optimizer state"""
        torch.save(self.target_network.state_dict(), filename)
        print(f"CNN 4-Channel Model saved to {filename}")
    
    def load_model(self, filename='cnn_dqn_model_pytorch.pth', learning_rate=0.0015):
        """Load model weights and optimizer state"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            self.q_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(checkpoint)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            print(f"CNN 4-Channel Model loaded from {filename}")
        else:
            print(f"CNN 4-Channel Model file {filename} not found, starting with random weights")

def create_dummy_4channel_state():
    """Create a dummy 4-channel image state for testing"""
    # Create a simple test 4-channel image (4, 84, 84)
    image = np.random.rand(4, 84, 84).astype(np.float32)
    return image 