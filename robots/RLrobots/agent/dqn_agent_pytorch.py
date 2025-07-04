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

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=11, action_size=8, learning_rate=0.015, epsilon=1, 
                 epsilon_min=0.01, epsilon_decay=0.999999, memory_size=50_000, device='auto'):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Size of state space (11 for robot state)
            action_size: Size of action space (8 for new RobotAction actions)
            learning_rate: Learning rate for optimizer
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            memory_size: Size of replay memory
            device: Device to run on ('cpu' or 'cuda' or 'auto')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.update_target_frequency = 50
        self.update_counter = 0
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = DQNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate )
        
        # Initialize target network
        self.update_target_network()
        
        print(f"DQN Agent initialized with {action_size} actions on {self.device}")
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_q_values(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values
    
    def get_optimal_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.target_network(state_tensor)
        return q_values.argmax().item()
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([experience[0] for experience in batch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in batch]).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in batch]).to(self.device)
        next_states = torch.FloatTensor([experience[3] for experience in batch]).to(self.device)
        dones = torch.IntTensor([experience[4] for experience in batch]).to(self.device)
        
        # Current Q values - fix the gather operation
        q_values = self.q_network(states).squeeze(1)  # Shape: (batch_size, action_size)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)
        
        # Next Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(2)[0]  # Shape: (batch_size,)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and update
        # print("learning with epsilon", self.epsilon)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filename='dqn_model_pytorch.pth'):
        """Save model weights and optimizer state"""
        torch.save(self.target_network.state_dict(), filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='dqn_model_pytorch.pth', learning_rate=0.0005):
        """Load model weights and optimizer state"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            self.q_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(checkpoint)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} not found, starting with random weights")



def normalize_state(state_dict):
    """Normalize state values for better neural network performance"""
    
    # Discretize continuous values for better learning
    x_discrete = int(state_dict.get('x', 0) / 50)  # 8 discrete x positions
    y_discrete = int(state_dict.get('y', 0) / 50)  # 8 discrete y positions
    heading_discrete = int(state_dict.get('heading', 0) / 45)  # 8 discrete heading directions
    gun_heading_discrete = int(state_dict.get('gunHeading', 0) / 45)  # 8 discrete gun directions
    gun_heat_discrete = int(state_dict.get('gunHeat', 0))  # 3 discrete gun heat levels (0, 1, 2+)
    energy_discrete = int(state_dict.get('energy', 0) / 10 )  # 10 discrete energy levels
    enemy_distance_discrete =  int(state_dict.get('enemyDistance', 0) / 50 )

    state = np.array([
        x_discrete / 8,  # Discrete x position
        y_discrete / 8,  # Discrete y position
        energy_discrete / 10,  # Normalize energy
        gun_heat_discrete / 3,  # Discrete gun heat
        enemy_distance_discrete / 12,  # Normalize enemy distance
        state_dict.get('distanceRemaining', 0) / 100,  
        state_dict.get('gunOnTarget', 0),  # Normalize heading
        state_dict.get('radarOnTarget', 0) ,  # Normalize heading

        # state_dict.get('heading', 0) / 360,  # Normalize heading
        # state_dict.get('gunHeading', 0) / 360,  # Normalize gun heading
        # state_dict.get('velocity', 0) / 8,  # Normalize velocity

        # state_dict.get('enemyBearing', 0) / 180,  # Normalize enemy bearing
        # state_dict.get('enemyHeading', 0) / 360,  # Normalize enemy heading
        # state_dict.get('enemyVelocity', 0) / 8,  # Normalize enemy velocity
        # state_dict.get('enemyX', 0) / 400,  # Normalize enemy x
        # state_dict.get('enemyY', 0) / 400,  # Normalize enemy y
    ])
    return state.reshape(1, -1)

# def calculate_reward(current_state, next_state, action, hit_enemy=False, hit_by_enemy=False, won=False, lost=False):
#     """Calculate reward based on state changes and events"""
#     reward = 0
    
#     # Base reward for staying alive
#     reward += 0.1
    
#     # Check for events in the state message
#     if 'events' in next_state:
#         events = next_state['events']
#         hit_enemy = events.get('hitEnemy', False)
#         hit_by_enemy = events.get('hitByEnemy', False)
#         won = events.get('won', False)
#         lost = events.get('lost', False)
#         bullet_hit_bullet = events.get('bulletHitBullet', False)
#         hit_robot = events.get('hitRobot', False)
#         bullet_missed = events.get('bulletMissed', False)
#         hit_wall = events.get('hitWall', False)
#         energy_change = events.get('energyChange', 0)
        
#         # Reward for energy gain (hitting enemy)
#         if hit_enemy:
#             reward += 10
#             print(f"Reward for hitting enemy: +10")
        
#         # Penalty for energy loss (being hit)
#         if hit_by_enemy:
#             reward -= 5
#             print(f"Penalty for being hit: -5")
        
#         # Penalty for bullet hitting bullet (wasted shot)
#         if bullet_hit_bullet:
#             reward -= 2
#             print(f"Penalty for bullet hit bullet: -2")
        
#         # Penalty for hitting robot (collision)
#         if hit_robot:
#             reward -= 3
#             print(f"Penalty for hitting robot: -3")
        
#         # Penalty for missing shots
#         if bullet_missed:
#             reward -= 1
#             print(f"Penalty for bullet missed: -1")
        
#         # Penalty for hitting wall
#         if hit_wall:
#             reward -= 2
#             print(f"Penalty for hitting wall: -2")
        
#         # Reward for winning
#         if won:
#             reward += 100
#             print(f"Reward for winning: +100")
        
#         # Penalty for losing
#         if lost:
#             reward -= 50
#             print(f"Penalty for losing: -50")
        
#         # Reward for energy change
#         if energy_change > 0:
#             reward += energy_change * 2  # Bonus for energy gain
#         elif energy_change < 0:
#             reward += energy_change  # Penalty for energy loss
    
#     # Reward for being close to enemy (encourages engagement)
#     enemy_distance = next_state.get('enemyDistance', 1000)
#     if enemy_distance < 200:
#         reward += 2
#     elif enemy_distance > 500:
#         reward -= 1
    
#     # Reward for having high energy
#     energy = next_state.get('energy', 0)
#     if energy > 80:
#         reward += 1
#     elif energy < 20:
#         reward -= 2
    
#     # Reward for action efficiency
#     if action == 1 and next_state.get('gunHeat', 0) == 0:  # Firing when gun is ready
#         reward += 1
#     elif action == 1 and next_state.get('gunHeat', 0) > 0:  # Firing when gun is hot
#         reward -= 2
    
#     return reward 