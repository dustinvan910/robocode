#!/usr/bin/env python3
"""
Main script for PPO Robocode robot
Acts as a WebSocket server that Robocode can connect to
"""

import json
import time
import numpy as np
from ppo_agent import PPO_agent
from action_converter import action_adapter
import os
import sys
import threading
from collections import deque
import asyncio
import websockets
import logging
import traceback
import argparse
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PPO Agent hyperparameters
state_dim = 11  # State dimension from robot
action_dim = 8  # Number of possible actions
net_width = 64  # Network width
T_horizon = 2048  # Trajectory horizon
gamma = 0.99  # Discount factor
lambd = 0.95  # GAE lambda
clip_rate = 0.2  # PPO clip rate
K_epochs = 10  # Number of epochs for PPO update
a_optim_batch_size = 64  # Actor optimizer batch size
c_optim_batch_size = 64  # Critic optimizer batch size
a_lr = 3e-4  # Actor learning rate
c_lr = 1e-3  # Critic learning rate
entropy_coef = 0.01  # Entropy coefficient
entropy_coef_decay = 0.995  # Entropy decay
l2_reg = 1e-3  # L2 regularization
Distribution = 'GS_m'  # Distribution type: 'Beta', 'GS_ms', 'GS_m'
dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize PPO agent
agent = PPO_agent(
    state_dim=state_dim,
    action_dim=action_dim,
    net_width=net_width,
    T_horizon=T_horizon,
    gamma=gamma,
    lambd=lambd,
    clip_rate=clip_rate,
    K_epochs=K_epochs,
    a_optim_batch_size=a_optim_batch_size,
    c_optim_batch_size=c_optim_batch_size,
    a_lr=a_lr,
    c_lr=c_lr,
    entropy_coef=entropy_coef,
    entropy_coef_decay=entropy_coef_decay,
    l2_reg=l2_reg,
    Distribution=Distribution,
    dvc=dvc
)

def normalize_state(state_data):
    """
    Normalize state data for the neural network
    Args:
        state_data: Dictionary containing robot state
    Returns:
        Normalized state array
    """
    # Extract relevant state features
    state = []
    
    # Robot position and orientation
    state.append(state_data.get('x', 0) / 800)  # Normalize x position
    state.append(state_data.get('y', 0) / 600)  # Normalize y position
    state.append(state_data.get('heading', 0) / 360)  # Normalize heading
    state.append(state_data.get('energy', 100) / 100)  # Normalize energy
    
    # Gun state
    state.append(state_data.get('gunHeading', 0) / 360)  # Normalize gun heading
    state.append(state_data.get('gunHeat', 0) / 3)  # Normalize gun heat
    
    # Movement
    state.append(state_data.get('velocity', 0) / 8)  # Normalize velocity
    state.append(state_data.get('distanceRemaining', 0) / 100)  # Normalize distance
    
    # Enemy information
    state.append(state_data.get('enemyDistance', 0) / 1000)  # Normalize enemy distance
    state.append(state_data.get('enemyBearing', 0) / 360)  # Normalize enemy bearing
    state.append(state_data.get('enemyVelocity', 0) / 8)  # Normalize enemy velocity
    
    return np.array(state, dtype=np.float32)

# Global state tracking
current_state = None
previous_state = None
previous_action = None
episode_reward = 0
episode_count = 0
total_reward = 0
training_losses = []
step_count = 0
win_episode = 0
death_episode = 0
play_time = 0
optimal_actions = []
episode_started = False
trajectory_idx = 0

# Training statistics
training_stats = {
    'episodes': [],
    'rewards': [],
    'losses': [],
    'wins': [],
    'deaths': [],
    'play_time': [],
    'learning_rate': [],
    'optimal_actions': [],
    'policy_losses': [],
    'value_losses': [],
    'entropy_losses': []
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Robot PPO Training')
parser.add_argument('--play', type=bool, default=False, help='Play real robot')
parser.add_argument('--continuous', type=bool, default=False, help='Continuous learning without episodes')
args = parser.parse_args()
no_learning = False

def start_episode():
    """Start a new episode"""
    global episode_reward, step_count, optimal_actions, episode_started, trajectory_idx
    episode_reward = 0
    step_count = 0
    optimal_actions = []
    episode_started = True
    trajectory_idx = 0
    
    logger.info(f"Episode {episode_count + 1} started")

def end_episode():
    """End current episode and update statistics"""
    global episode_count, total_reward, episode_reward, previous_state, previous_action, episode_started
    
    if not episode_started:
        return
    
    episode_count += 1
    total_reward += episode_reward
    
    training_stats['episodes'].append(episode_count)
    training_stats['rewards'].append(episode_reward)
    training_stats['learning_rate'].append(a_lr)
    training_stats['wins'].append(win_episode)
    training_stats['deaths'].append(death_episode)
    training_stats['play_time'].append(play_time)

    # Count optimal actions by type
    optimal_action_counts = {}
    for action in optimal_actions:
        optimal_action_counts[action] = optimal_action_counts.get(action, 0) + 1
    training_stats['optimal_actions'].append(optimal_action_counts)

    # Add average loss for this episode
    if training_losses:
        avg_loss = np.mean(training_losses[-50:])  # Last 50 losses
        training_stats['losses'].append(avg_loss)
    
    print(f"Episode {episode_count} ended with reward: {episode_reward:.2f}, play_time: {play_time}")
    
    # Reset episode variables
    episode_reward = 0
    previous_state = None
    previous_action = None
    episode_started = False

    # Save model periodically
    if episode_count % 200 == 0:
        print("Saving models")
        agent.save("Robocode", episode_count)
        save_training_stats()

def save_training_stats():
    """Save training statistics to file"""
    with open('ppo_training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    print("PPO training stats saved")

async def handle_robot(websocket):
    global current_state, previous_state, previous_action, episode_reward, episode_count, total_reward, training_losses, step_count, win_episode, death_episode, play_time, no_learning, optimal_actions, episode_started, trajectory_idx
    
    try:
        print("Robot connected to PPO server")
        async for message in websocket:
            try:
                state_data = json.loads(message)
                current_state = state_data
                
                if 'play' in state_data:
                    no_learning = state_data.get('play', 0)
                    continue
                
                if 'test' in state_data:
                    # For PPO, we can return action probabilities and value
                    normalized_state = normalize_state(state_data)
                    action, logprob_a = agent.select_action(normalized_state, deterministic=False)
                    # Get value estimate
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(normalized_state.reshape(1, -1)).to(dvc)
                        value = agent.critic(state_tensor).cpu().numpy()[0][0]
                    test_result = action_adapter(action, logprob_a, value, state_data)
                    await websocket.send(json.dumps(test_result))
                    return     
                           
                if 'gameStart' in state_data:
                    start_episode()
                    continue

                # Handle episode end
                if 'isWin' in state_data:
                    if args.play == True or no_learning == True:
                        return
                    
                    play_time = state_data.get('time')
                    reward = 0
                    if state_data.get('isWin'):
                        playtime_quant = (100 - play_time) / 100
                        reward = 1000 
                        win_episode += 1
                    else:
                        reward = -1000
                        death_episode += 1
                    
                    # Store final experience for PPO
                    if previous_state is not None and previous_action is not None:
                        agent.put_data(
                            normalize_state(previous_state),
                            previous_action,
                            reward,
                            normalize_state(current_state),
                            previous_action,  # logprob will be computed during training
                            1,  # done
                            1,  # dw (done without truncation)
                            trajectory_idx
                        )
                        trajectory_idx += 1
                        
                        # Train if trajectory is full
                        if trajectory_idx >= T_horizon:
                            agent.train()
                            trajectory_idx = 0
                    
                    episode_reward += reward
                    end_episode()
                    return

                # Check if state_data contains 'test' key and run get_optimal_action if it does
                if args.play == True or no_learning == True:
                    normalized_state = normalize_state(state_data)
                    action, logprob_a = agent.select_action(normalized_state, deterministic=True)
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(normalized_state.reshape(1, -1)).to(dvc)
                        value = agent.critic(state_tensor).cpu().numpy()[0][0]
                    action_response = action_adapter(action, logprob_a, value, state_data)
                    await websocket.send(json.dumps(action_response))
                    continue
                
                # Normalize state for neural network
                normalized_state = normalize_state(state_data)
                
                # Get action from PPO agent
                action, logprob_a = agent.select_action(normalized_state, deterministic=False)
                
                # Get value estimate
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(normalized_state.reshape(1, -1)).to(dvc)
                    value = agent.critic(state_tensor).cpu().numpy()[0][0]
                
                
                
                # Calculate reward if we have previous state
                if previous_state is not None and previous_action is not None:
                    reward = state_data.get('reward', 0) 
                    episode_reward += reward
                    
                    # Store experience in agent's trajectory
                    agent.put_data(
                        normalize_state(previous_state),
                        previous_action,
                        reward,
                        normalized_state,
                        logprob_a,
                        0,  # not done
                        0,  # not dw
                        trajectory_idx
                    )
                    trajectory_idx += 1
                    step_count += 1
                    
                    # Train the agent when trajectory is full
                    if trajectory_idx >= T_horizon:
                        agent.train()
                        trajectory_idx = 0
                
                # Send action back to robot with values
                action_response = action_adapter(action, logprob_a, value, state_data)
                # Record optimal action for this episode
                optimal_actions.append(action_response['action'])

                await websocket.send(json.dumps(action_response))
                
                # Update tracking variables
                previous_state = current_state.copy()
                previous_action = action

                            
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Error processing message: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                continue
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        print("Client disconnected. Final cleanup")
        # End episode if not in continuous mode
        end_episode()

async def main():
    server = await websockets.serve(handle_robot, "localhost", 5000)
    print("PyTorch PPO Robot server started on ws://localhost:5000")
    print(f"Using device: {dvc}")
    print(f"PPO hyperparameters:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Network width: {net_width}")
    print(f"  Trajectory horizon: {T_horizon}")
    print(f"  Actor learning rate: {a_lr}")
    print(f"  Critic learning rate: {c_lr}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE lambda: {lambd}")
    print(f"  Clip rate: {clip_rate}")
    print(f"  K epochs: {K_epochs}")
    print(f"  Distribution: {Distribution}")
    print(f"Mode: {'Continuous' if args.continuous else 'Episode-based'} learning")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main()) 