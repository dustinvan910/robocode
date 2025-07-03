import asyncio
import websockets
import json
import numpy as np
from dqn_agent_pytorch import DQNAgent, normalize_state

# Initialize PyTorch DQN agent
agent = DQNAgent(state_size=12, action_size=9, learning_rate=0.002)
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
    'learning_rate': []
}
import argparse 
import os
# Parse command line arguments
parser = argparse.ArgumentParser(description='Robot DQN Training')
parser.add_argument('--play', type=bool, default=False, help='Play real robot')
args = parser.parse_args()

async def handle_robot(websocket):
    global current_state, previous_state, previous_action, episode_reward, episode_count, total_reward, training_losses, step_count, win_episode, death_episode, play_time
    
    try:
        print("Robot connected")
        async for message in websocket:
            try:
                state_data = json.loads(message)
                current_state = state_data
                # print("state_data", state_data)
                # Check if state_data contains 'test' key and run get_qvalue if it does
                if args.play == True:
                    q_values = agent.get_optimal_action(normalize_state(state_data)).squeeze()
                    max_q_value = q_values.max().item()
                    await websocket.send(str(max_q_value))
                    continue
                
                if 'test' in state_data:
                    q_values = agent.get_q_values(normalize_state(state_data)).squeeze()
                    await websocket.send(str(q_values.tolist()))
                    return
                
                if 'isWin' in state_data:
                    play_time = state_data.get('time')
                    reward = 0
                    if state_data.get('isWin'):
                        playtime_quant = (100 - play_time) / 100
                        reward = 100 * (1 + playtime_quant)
                        win_episode += 1
                    else:
                        reward = -100
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
                
                # if state_data.get('reward', 0) >= 100 or state_data.get('reward', 0) <= -100:
                #     await websocket.close()
                #     print("Robot disconnected")
                #     return
                
                # # Normalize state for neural network
                normalized_state = normalize_state(state_data)
                
                # print(current_state)

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
                continue
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # print("Client disconnected. Final cleanup")
        # End episode
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
            # Add average loss for this episode
            if training_losses:
                avg_loss = np.mean(training_losses[-50:])  # Last 50 losses
                training_stats['losses'].append(avg_loss)
            
            print(f"Episode {episode_count} ended with reward: {episode_reward:.2f}, play_time: {play_time}")
            
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