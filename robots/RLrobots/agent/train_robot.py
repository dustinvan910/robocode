#!/usr/bin/env python3
"""
Training script for the DQN Robocode robot
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from dqn_agent import DQNAgent
import os

def plot_training_progress():
    """Plot training statistics"""
    if not os.path.exists('training_stats.json'):
        print("No training stats found. Run the robot first to generate data.")
        return
    
    with open('training_stats.json', 'r') as f:
        stats = json.load(f)
    
    if not stats['episodes']:
        print("No training data available yet.")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot rewards
    ax1.plot(stats['episodes'], stats['rewards'], 'b-', alpha=0.6)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot average rewards (moving average)
    if len(stats['rewards']) > 10:
        window = min(10, len(stats['rewards']) // 10)
        avg_rewards = np.convolve(stats['rewards'], np.ones(window)/window, mode='valid')
        episodes_avg = stats['episodes'][window-1:]
        ax1.plot(episodes_avg, avg_rewards, 'r-', linewidth=2, label=f'{window}-episode average')
        ax1.legend()
    
    # Plot epsilon
    ax2.plot(stats['episodes'], stats['epsilon'], 'g-')
    ax2.set_title('Epsilon (Exploration Rate)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True)
    
    # Plot win/loss ratio
    if 'wins' in stats and 'losses' in stats:
        total_games = stats['wins'] + stats['losses']
        if total_games > 0:
            win_rate = stats['wins'] / total_games * 100
            ax3.bar(['Win Rate'], [win_rate], color='green', alpha=0.7)
            ax3.set_title(f'Win Rate: {win_rate:.1f}% ({stats["wins"]}/{total_games})')
            ax3.set_ylabel('Win Rate (%)')
            ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_training_summary():
    """Print current training statistics"""
    if not os.path.exists('training_stats.json'):
        print("No training stats found.")
        return
    
    with open('training_stats.json', 'r') as f:
        stats = json.load(f)
    
    if not stats['episodes']:
        print("No training data available yet.")
        return
    
    print("\n=== TRAINING SUMMARY ===")
    print(f"Total Episodes: {len(stats['episodes'])}")
    print(f"Latest Episode: {stats['episodes'][-1]}")
    print(f"Average Reward: {np.mean(stats['rewards']):.2f}")
    print(f"Best Reward: {max(stats['rewards']):.2f}")
    print(f"Current Epsilon: {stats['epsilon'][-1]:.3f}")
    
    if len(stats['rewards']) > 10:
        recent_avg = np.mean(stats['rewards'][-10:])
        print(f"Recent 10-episode Average: {recent_avg:.2f}")
    
    if 'wins' in stats and 'losses' in stats:
        total_games = stats['wins'] + stats['losses']
        if total_games > 0:
            win_rate = stats['wins'] / total_games * 100
            print(f"Win Rate: {win_rate:.1f}% ({stats['wins']}/{total_games})")

def reset_training():
    """Reset training progress"""
    if os.path.exists('training_stats.json'):
        os.remove('training_stats.json')
        print("Training stats reset.")
    
    if os.path.exists('dqn_model.pkl'):
        os.remove('dqn_model.pkl')
        print("Model reset.")

def main():
    print("DQN Robocode Training Monitor")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. View training progress")
        print("2. Print training summary")
        print("3. Reset training")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            plot_training_progress()
        elif choice == '2':
            print_training_summary()
        elif choice == '3':
            confirm = input("Are you sure you want to reset all training progress? (y/n): ").strip().lower()
            if confirm == 'y':
                reset_training()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 