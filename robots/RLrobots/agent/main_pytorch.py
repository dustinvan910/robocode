import asyncio
import websockets
import json
import numpy as np
from dqn_agent_pytorch import DQNAgent, normalize_state
import traceback

# Initialize PyTorch DQN agent
agent = DQNAgent(state_size=8, action_size=8, learning_rate=0.002)
agent.load_model(learning_rate=0.002, filename='dqn_model_pytorch.pth')  # Load existing model if available

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

# State statistics tracking
state_stats = {
    'x': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'y': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'heading': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'energy': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'gunHeading': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'gunHeat': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'velocity': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'distanceRemaining': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'enemyBearing': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'enemyDistance': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'enemyHeading': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'enemyX': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'enemyY': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'enemyVelocity': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'reward': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'gunOnTarget': {'min': float('inf'), 'max': float('-inf'), 'count': 0},
    'radarOnTarget': {'min': float('inf'), 'max': float('-inf'), 'count': 0}
}

def update_state_stats(state_data):
    """Update min/max statistics for each state variable"""
    for key, value in state_data.items():
        if key in state_stats and isinstance(value, (int, float)):
            if value < state_stats[key]['min']:
                state_stats[key]['min'] = value
            if value > state_stats[key]['max']:
                state_stats[key]['max'] = value
            state_stats[key]['count'] += 1

def print_state_stats():
    """Print current state statistics"""
    print("\n=== State Statistics ===")
    for key, stats in state_stats.items():
        if stats['count'] > 0:
            print(f"{key}: min={stats['min']:.2f}, max={stats['max']:.2f}, count={stats['count']}")
    print("=======================\n")

def save_state_stats():
    """Save state statistics to file"""
    with open('state_stats.json', 'w') as f:
        json.dump(state_stats, f, indent=2)
    print("State statistics saved to state_stats.json")

# Training statistics
training_stats = {
    'episodes': [],
    'rewards': [],
    'epsilon': [],
    'losses': [],
    'wins': [],
    'deaths': [],
    'play_time': [],
    'losses': [],
    'learning_rate': [],
    'optimal_actions': []
}
import argparse 
import os
# Parse command line arguments
parser = argparse.ArgumentParser(description='Robot DQN Training')
parser.add_argument('--play', type=bool, default=False, help='Play real robot')
args = parser.parse_args()
no_learning = False

async def handle_robot(websocket):
    global current_state, previous_state, previous_action, episode_reward, episode_count, total_reward, training_losses, step_count, win_episode, death_episode, play_time, no_learning, optimal_actions
    
    try:
        print("Robot connected 3")
        async for message in websocket:
            try:
                state_data = json.loads(message)
                current_state = state_data
                
                # Update state statistics
                update_state_stats(state_data)
                
                if 'play' in state_data:
                    no_learning = state_data.get('play', 0)
                    continue
                
                if 'test' in state_data:
                    q_values = agent.get_q_values(normalize_state(state_data)).squeeze()
                    await websocket.send(str(q_values.tolist()))
                    return
                
                

                if 'isWin' in state_data:
                    if args.play == True or no_learning == True:
                        return
                    play_time = state_data.get('time')
                    reward = 0
                    if state_data.get('isWin'):
                        reward = 1000
                        win_episode += 1
                    else:
                        reward = -1000
                        death_episode += 1
                    
                    agent.remember(
                        normalize_state(previous_state),
                        previous_action,
                        reward,
                        normalized_state,
                        1
                    )
                    episode_reward+=reward
                    return

                # Check if state_data contains 'test' key and run get_qvalue if it does
                if args.play == True or no_learning == True:
                    action = agent.get_optimal_action(normalize_state(state_data))
                    await websocket.send(str(action))
                    continue
                
                # # Normalize state for neural network
                normalized_state = normalize_state(state_data)
                
                opt_action = agent.get_optimal_action(normalized_state)
                
                # Record optimal action for this episode
                optimal_actions.append(opt_action)
                

                # Get action from DQN agent
                action = agent.act(normalized_state)
                # print(current_state, action)

                # Calculate reward if we have previous state
                if previous_state is not None and previous_action is not None:
                    reward = state_data.get('reward', 0) 
                    episode_reward += reward
                    
                    # Store experience in agent's memory
                    agent.remember(
                        normalize_state(previous_state),
                        previous_action,
                        reward,
                        normalized_state,
                        0  # Not done yet
                    )
                    step_count += 1                    
                    # Train the agent and get loss
                    if step_count % 4 == 0:
                        loss = agent.replay()
                        if loss is not None:
                            training_losses.append(loss)
                
                # Send action back to robot
                await websocket.send(str(action))
                
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
        # End episode
        if args.play == True or no_learning == True:
            return

        if previous_state is not None and previous_action is not None:            
            # Update statistics
            episode_count += 1
            total_reward += episode_reward
            
            training_stats['episodes'].append(episode_count)
            training_stats['rewards'].append(episode_reward)
            training_stats['epsilon'].append(agent.epsilon)
            training_stats['learning_rate'].append(agent.learning_rate)
            training_stats['wins'].append(win_episode)
            training_stats['deaths'].append(death_episode)
            training_stats['play_time'].append(play_time)

            # Count optimal actions by type
            optimal_action_counts = {}
            for action in optimal_actions:
                optimal_action_counts[action] = optimal_action_counts.get(action, 0) + 1
            training_stats['optimal_actions'].append(optimal_action_counts)
            optimal_actions = []

            # Add average loss for this episode
            if training_losses:
                avg_loss = np.mean(training_losses[-50:])  # Last 50 losses
                training_stats['losses'].append(avg_loss)
            
            print(f"Episode {episode_count} ended with reward: {episode_reward:.2f}, play_time: {play_time}")
            print(f"Optimal actions: {optimal_action_counts}")
            
            # Print state statistics every 10 episodes
            # if episode_count % 10 == 0:
            #     print_state_stats()
            
            # Reset episode variables
            episode_reward = 0
            previous_state = None
            previous_action = None

            # Save model periodically
            if episode_count % 10 == 0:
                agent.save_model()
                save_training_stats()

def save_training_stats():
    """Save training statistics to file"""
    with open('training_stats_pytorch.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    print("Training stats saved")

async def main():
    server = await websockets.serve(handle_robot, "localhost", 5000)
    print("PyTorch DQN Robot server started on ws://localhost:5000")
    print(f"Agent epsilon: {agent.epsilon}")
    print(f"Using device: {agent.device}")
    await server.wait_closed()



if __name__ == "__main__":
    asyncio.run(main()) 