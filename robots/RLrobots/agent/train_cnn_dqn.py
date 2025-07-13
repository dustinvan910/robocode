#!/usr/bin/env python3
"""
CNN DQN Training Script for Robocode

This script provides an easy way to train the CNN DQN agent with different parameters.
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Train CNN DQN Agent for Robocode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.999999, help='Epsilon decay rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--memory-size', type=int, default=50000, help='Replay memory size')
    parser.add_argument('--play', action='store_true', help='Run in play mode (no learning)')
    parser.add_argument('--port', type=int, default=5000, help='WebSocket port')
    
    args = parser.parse_args()
    
    print("=== CNN DQN Training Configuration ===")
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Initial Epsilon: {args.epsilon}")
    print(f"Epsilon Decay: {args.epsilon_decay}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Memory Size: {args.memory_size}")
    print(f"Play Mode: {args.play}")
    print(f"Port: {args.port}")
    print("=====================================")
    
    # Check if required files exist
    required_files = [
        'cnn_dqn_agent_pytorch.py',
        'main_cnn_pytorch.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file {file} not found!")
            sys.exit(1)
    
    # Start the training
    print("\nStarting CNN DQN training...")
    print("Make sure your Robocode robot is configured to connect to this server.")
    print("Press Ctrl+C to stop training.\n")
    
    try:
        # Run the main CNN training script
        cmd = [sys.executable, 'main_cnn_pytorch.py']
        if args.play:
            cmd.extend(['--play', 'True'])
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error running training script: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 