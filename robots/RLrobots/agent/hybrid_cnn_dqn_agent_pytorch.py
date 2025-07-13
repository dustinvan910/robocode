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


class CNNNetwork(nn.Module):
    def __init__(self, image_channels=4, hidden_size=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(image_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class StateNetwork(nn.Module):
    def __init__(self, state_size=13, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class HybridNetwork(nn.Module):
    def __init__(self, cnn_hidden_size=512, state_hidden_size=128, combined_hidden_size=256, action_size=8):
        super().__init__()
        self.cnn_network = CNNNetwork(hidden_size=cnn_hidden_size)
        self.state_network = StateNetwork(hidden_size=state_hidden_size)
        
        # Combined network that processes both CNN and state features
        self.combined_network = nn.Sequential(
            nn.Linear(cnn_hidden_size + state_hidden_size, combined_hidden_size),
            nn.ReLU(),
            nn.Linear(combined_hidden_size, combined_hidden_size),
            nn.ReLU(),
            nn.Linear(combined_hidden_size, action_size),
        )

    def forward(self, image_input, state_input):
        cnn_features = self.cnn_network(image_input)
        state_features = self.state_network(state_input)
        
        # Concatenate features from both networks
        combined_features = torch.cat([cnn_features, state_features], dim=1)
        
        # Process through combined network
        q_values = self.combined_network(combined_features)
        return q_values


class HybridReplayBuffer:
    """Custom replay buffer for hybrid observations (image + state)"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state_image, state_vector, action, reward, next_state_image, next_state_vector, done):
        """Store a transition in the replay buffer"""
        self.buffer.append({
            'state_image': state_image,
            'state_vector': state_vector,
            'action': action,
            'reward': reward,
            'next_state_image': next_state_image,
            'next_state_vector': next_state_vector,
            'done': done
        })
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        
        # Extract components
        state_images = torch.FloatTensor(np.array([exp['state_image'] for exp in batch]))
        state_vectors = torch.FloatTensor(np.array([exp['state_vector'] for exp in batch]))
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_state_images = torch.FloatTensor(np.array([exp['next_state_image'] for exp in batch]))
        next_state_vectors = torch.FloatTensor(np.array([exp['next_state_vector'] for exp in batch]))
        dones = torch.FloatTensor([exp['done'] for exp in batch])
        
        return {
            'state_images': state_images,
            'state_vectors': state_vectors,
            'actions': actions,
            'rewards': rewards,
            'next_state_images': next_state_images,
            'next_state_vectors': next_state_vectors,
            'dones': dones
        }
    
    def __len__(self):
        return len(self.buffer)


class HybridCNNDQNAgent:
    def __init__(self, image_channels=4, image_size=128, state_size=13, action_size=8, 
                 learning_rate=0.0015, epsilon_max=1, epsilon_min=0.01, epsilon_decay=0.9, 
                 memory_size=20_000, device='auto'):
        """
        Initialize Hybrid CNN DQN Agent that combines image and state vector processing
        
        Args:
            image_channels: Number of image channels (4 for frame buffer)
            image_size: Size of input images
            state_size: Size of state vector
            action_size: Size of action space (8 for new RobotAction actions)
            learning_rate: Learning rate for optimizer
            epsilon_max: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            memory_size: Size of replay memory
            device: Device to run on ('cpu' or 'cuda' or 'auto')
        """
        self.image_channels = image_channels
        self.image_size = image_size
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        self.batch_size = 32
        self.gamma = 0.99
        self.update_target_frequency = 100
        self.update_counter = 0
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Custom replay buffer for hybrid observations
        self.memory = HybridReplayBuffer(memory_size)

        # Networks
        self.q_network = HybridNetwork(action_size=action_size).to(self.device)
        self.target_network = HybridNetwork(action_size=action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        print(f"Hybrid CNN DQN Agent initialized with {action_size} actions on {self.device}")
        print(f"Image channels: {image_channels}, State size: {state_size}")

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state_image, state_vector, action, reward, next_state_image, next_state_vector, done):
        """Store experience in replay memory"""
        self.update_counter += 1
        self.memory.push(state_image, state_vector, action, reward, next_state_image, next_state_vector, done)
    
    def act(self, state_image, state_vector):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_image_tensor = torch.FloatTensor(state_image).unsqueeze(0).to(self.device)
        state_vector_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_image_tensor, state_vector_tensor)
        return q_values.argmax().item()
    
    def get_optimal_action(self, state_image, state_vector):
        """Get optimal action without exploration (epsilon=0)"""
        state_image_tensor = torch.FloatTensor(state_image).to(self.device)
        state_vector_tensor = torch.FloatTensor(state_vector).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_image_tensor, state_vector_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        start_learning = self.batch_size 
        if self.update_counter < start_learning or len(self.memory) < self.batch_size:
            return
        
        t1 = time.time()
        
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        
        # Move data to device
        state_images = batch['state_images'].to(self.device)
        state_vectors = batch['state_vectors'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_state_images = batch['next_state_images'].to(self.device)
        next_state_vectors = batch['next_state_vectors'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_images, state_vectors)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_state_images, next_state_vectors)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        t3 = time.time()
        print(f"Time taken to update optimizer: {t3 - t1}", self.update_counter, self.epsilon)
        
        # Update target network
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = self.linear_schedule(self.epsilon_max, self.epsilon_min, 1_000_000, self.update_counter - start_learning)
        
        return loss.item()
    
    def save_model(self, filename='hybrid_cnn_dqn_model_pytorch.pth'):
        """Save model weights and optimizer state"""
        torch.save(self.target_network.state_dict(), filename)
        print(f"Hybrid CNN Model saved to {filename}")
    
    def load_model(self, filename='hybrid_cnn_dqn_model_pytorch.pth', learning_rate=0.0015):
        """Load model weights and optimizer state"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            self.q_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(checkpoint)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            print(f"Hybrid CNN Model loaded from {filename}")
        else:
            print(f"Hybrid CNN Model file {filename} not found, starting with random weights")


def normalize_state(state_dict):
    """Normalize state values for better neural network performance"""
    
    state = np.array([
        state_dict.get('x', 0) / 400,  # Normalize x position
        state_dict.get('y', 0) / 400,  # Normalize y position
        state_dict.get('heading', 0) / 360,  # Normalize heading
        state_dict.get('energy', 0) / 100,  # Normalize energy
        state_dict.get('gunHeat', 0) / 3,  # Normalize gun heat
        state_dict.get('gunHeading', 0) / 360,  # Normalize gun heading
        state_dict.get('velocity', 0) / 8,  # Normalize velocity
        state_dict.get('distanceRemaining', 0) / 100,  # Normalize distance remaining
        state_dict.get('enemyBearing', 0) / 180,  # Normalize enemy bearing
        state_dict.get('enemyDistance', 0) / 400,  # Normalize enemy distance
        state_dict.get('enemyHeading', 0) / 360,  # Normalize enemy heading
        state_dict.get('enemyVelocity', 0) / 8,  # Normalize enemy velocity
        state_dict.get('turnRemaining', 0) / 360,  # Normalize turn remaining
        # state_dict.get('enemyX', 0) / 400,  # Normalize enemy x
        # state_dict.get('enemyY', 0) / 400,  # Normalize enemy y
    ])
    
    return state