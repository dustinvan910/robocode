import asyncio
import websockets
import json
import numpy as np
from cnn_dqn_agent_pytorch import CNNDQNAgent
import traceback
from PIL import Image

# Initialize PyTorch CNN DQN agent for grayscale image input
agent = CNNDQNAgent(image_channels=4, action_size=7, learning_rate=0.0001)
agent.load_model(learning_rate=0.0001, filename='cnn_dqn_model_pytorch.pth')  # Load existing model if available

# Global state tracking
current_state_image = None
previous_state_image = None
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
from collections import deque
frame_buffer = deque(maxlen=4)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Robot CNN DQN Training with Grayscale Image Input')
parser.add_argument('--play', type=bool, default=False, help='Play real robot')
args = parser.parse_args()
no_learning = False

def convert_frame_buffer_to_image_state(frame_buffer):
    """Convert frame buffer to 4-channel image state"""
    if len(frame_buffer) != 4:
        return None
    
    # Stack the 4 frames into a single 4-channel image
    # Each frame is (1, 84, 84), so result will be (1, 4, 84, 84)
    stacked_frames = np.concatenate(list(frame_buffer), axis=0)
    stacked_frames = np.expand_dims(stacked_frames, axis=0)
    
    return stacked_frames

async def handle_robot(websocket):
    global current_state_image, previous_state_image, previous_action, episode_reward, episode_count, total_reward, training_losses, step_count, win_episode, death_episode, play_time, no_learning, optimal_actions, frame_buffer
    
    try:
        print("CNN Grayscale Robot connected")
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Convert raw grayscale image bytes to numpy array
                    # print("Binary message received")

                    # Assuming the image is sent as raw grayscale bytes
                    # You may need to adjust width and height based on your actual image dimensions
                    width, height = 400, 400  # Adjust these values based on your battle view size
                    
                    # Convert bytes to numpy array
                    image_array = np.frombuffer(message, dtype=np.uint8)
                    
                    # Reshape to 2D grayscale image
                    if len(image_array) == width * height:
                        image_2d = image_array.reshape((height, width))
                        # Convert to PIL Image, resize to 84x84, and save as PNG
                        image = Image.fromarray(image_2d, mode='L')
                        image = image.resize((84, 84))
                        image.save("image.png")
                        # Add channel dimension for CNN (1, 84, 84)
                        image_array = np.expand_dims(image, axis=0)
                        frame_buffer.append(image_array)
                    else:
                        print(f"Unexpected image data size: {len(image_array)} bytes")
                    continue

                # print("JSON message received")
                state_data = json.loads(message)
                
                if 'play' in state_data:
                    no_learning = state_data.get('play', 0)
                    continue
                
                # if 'test' in state_data:
                #     # Process grayscale image state for testing
                #     image_state = process_image_state(state_data)
                #     q_values = agent.get_q_values(image_state).squeeze()
                #     await websocket.send(str(q_values.tolist()))
                #     return
                
                # not enough frames in buffer, send 0 action
                if len(frame_buffer) != 4:
                    await websocket.send(str(0))
                    continue

                # Convert frame buffer to 4-channel image state
                image_state = convert_frame_buffer_to_image_state(frame_buffer)
                if image_state is None:
                    await websocket.send(str(0))
                    continue
                
                current_state_image = image_state
                
                if 'isWin' in state_data:
                    if args.play == True or no_learning == True:
                        return
                    play_time = state_data.get('time')
                    reward = 0
                    if state_data.get('isWin'):
                        reward = 100
                        win_episode += 1
                    else:
                        reward = -100
                        death_episode += 1
                    
                    agent.remember(
                        previous_state_image,
                        previous_action,
                        reward,
                        image_state,
                        1
                    )
                    episode_reward += reward
                    return

                # Check if state_data contains 'test' key and run get_qvalue if it does
                if args.play == True or no_learning == True:
                    action = agent.get_optimal_action(image_state)
                    await websocket.send(str(action))
                    continue
                
                # opt_action = agent.get_optimal_action(image_state)
                
                # # Record optimal action for this episode
                # optimal_actions.append(opt_action)
                
                # Get action from CNN DQN agent
                action = agent.act(image_state)

                # Calculate reward if we have previous state
                if previous_state_image is not None and previous_action is not None:
                    
                    reward = state_data.get('reward', 0) 
                    episode_reward += reward
                    # print("reward : ", reward)
                    # Store experience in agent's memory
                    agent.remember(
                        previous_state_image,
                        previous_action,
                        reward,
                        image_state,
                        0  # Not done yet
                    )
                    step_count += 1                    
                    # Train the agent and get loss
                    if step_count % 10 == 0:
                        loss = agent.replay()
                        if loss is not None:
                            training_losses.append(loss)
                
                # Send action back to robot
                await websocket.send(str(action))
                
                # Update tracking variables
                previous_state_image = image_state
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
        print("CNN Grayscale Client disconnected. Final cleanup")
        # End episode
        if args.play == True or no_learning == True:
            return

        if previous_state_image is not None and previous_action is not None:            
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
            
            print(f"CNN Grayscale Episode {episode_count} ended with reward: {episode_reward:.2f}, play_time: {play_time}")
            print(f"CNN Grayscale Optimal actions: {optimal_action_counts}")
            
            # Reset episode variables
            episode_reward = 0
            previous_state_image = None
            previous_action = None

            # Save model periodically
            if episode_count % 10 == 0:
                agent.save_model()
                save_training_stats()

def save_training_stats():
    """Save training statistics to file"""
    with open('cnn_grayscale_training_stats_pytorch.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    print("CNN Grayscale Training stats saved")

async def main():
    server = await websockets.serve(handle_robot, "localhost", 5000)
    print("PyTorch CNN Grayscale DQN Robot server started on ws://localhost:5000")
    print(f"CNN Grayscale Agent epsilon: {agent.epsilon}")
    print(f"CNN Grayscale Using device: {agent.device}")
    print(f"CNN Grayscale Input channels: {agent.image_channels}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main()) 