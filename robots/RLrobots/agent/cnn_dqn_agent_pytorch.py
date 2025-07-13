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
import time
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer


class CNNNetwork(nn.Module):
    def __init__(self, image_channels=4, action_size=8, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(image_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x):
        return self.network(x / 255.0)

class CNNDQNAgent:
    def __init__(self, image_channels=4, action_size=8, learning_rate=0.0015, epsilon=1, 
                 epsilon_min=0.01, epsilon_decay=0.99999, memory_size=20_000, device='auto'):
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
        
        self.batch_size = 32  # Reduced batch size for image processing
        self.gamma = 0.99  # Discount factor
        self.update_target_frequency = 50
        self.update_counter = 0
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Define observation and action spaces for ReplayBuffer
        # Observation space: 4-channel image (4, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(image_channels, 84, 84), dtype=np.uint8
        )
        
        # Action space: discrete actions (0 to action_size-1)
        self.action_space = gym.spaces.Discrete(action_size)
        
        self.memory = ReplayBuffer(
            memory_size,
            self.observation_space,
            self.action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False
        )

        # Networks
        self.q_network = CNNNetwork(image_channels, action_size).to(self.device)
        self.target_network = CNNNetwork(image_channels, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        print(f"CNN DQN Agent initialized with {action_size} actions on {self.device}")
        print(f"4-channel image channels: {image_channels}")
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state_image, action, reward, next_state_image, done):
        """Store experience in replay memory"""
        self.update_counter += 1
        action = np.array([action])
        reward = np.array([reward])
        done = np.array([done])
        self.memory.add(state_image,next_state_image, action, reward, done, [{} ])
    
    # def get_q_values(self, state_image):
    #     state_tensor = torch.FloatTensor(state_image).unsqueeze(0).to(self.device)
    #     q_values = self.q_network(state_tensor)
    #     return q_values
    
    # def get_optimal_action(self, state_image):
    #     state_tensor = torch.FloatTensor(state_image).unsqueeze(0).to(self.device)
    #     q_values = self.target_network(state_tensor)
    #     return q_values.argmax().item()
    
    def act(self, state_image):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state_image).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        

        if self.update_counter < self.batch_size:
            return
        
        t1 = time.time()
        # Sample batch from memory
        data = self.memory.sample(self.batch_size)
        
        # Next Q values (using target network)
        with torch.no_grad():
            target_max, _ = self.target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
            
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()



        t3 = time.time()
        print(f"Time taken to update optimizer: {t3 - t1}")
        # Update target network
        
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
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

def create_robot_observation_space(state_size=8):
    """Create observation space for robot state vector"""
    return gym.spaces.Box(
        low=0, high=1, shape=(state_size,), dtype=np.float32
    )

def create_robot_action_space(action_size=8):
    """Create action space for robot actions"""
    return gym.spaces.Discrete(action_size)

def create_image_observation_space(image_channels=4, height=84, width=84):
    """Create observation space for image input"""
    return gym.spaces.Box(
        low=0, high=255, shape=(image_channels, height, width), dtype=np.uint8
    ) 