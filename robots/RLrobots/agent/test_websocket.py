#!/usr/bin/env python3
"""
Test script to demonstrate WebSocket communication with Robocode robot
"""

import asyncio
import websockets
import json
import time


def json_to_state(json_data):
    """Convert JSON data to normalized state array for the neural network"""
    if isinstance(json_data, str):
        state_dict = json.loads(json_data)
    else:
        state_dict = json_data
    
    # Normalize state values (same as in dqn_agent_pytorch.py)
    state = [
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
    ]
    
    return state

async def test_robot_communication():
    """Test the WebSocket communication with the robot"""
    
    uri = "ws://localhost:5000"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to robot server")
            
            # Send a test message to trigger Q-value response
            test_state = {"play":True, 'x': 301.01, 'y': 173.9, 'heading': 114.36, 'energy': 102.2, 'gunHeading': 239.83, 'gunHeat': 0, 'velocity': 0.0, 'distanceRemaining': 0.0, 'enemyBearing': 0.0, 'enemyDistance': 100, 'enemyHeading': 0.0, 'enemyX': 0.0, 'enemyY': 0.0} 
            
            await websocket.send(json.dumps(test_state))
            print("Sent test state")
            
            # Receive the Q-values response
            q_values = await websocket.recv()
            q_values = json.loads(q_values)

            action_names = [
            "Do Nothing",
            "RunAwayLeft", 
            "RunAwayRight",
            "RunAwayBack",
            "RunAhead",
            "Fire1",
            "Fire2",
            "Fire3",
            "Aim"
            ]

            action_names = [
            "Do Nothing",
            "RunAwayLeft", 
            "RunAwayRight",
            "RunAwayBack",
            "RunAhead",
            "Fire1",
            "Aim"
            ]
            print("q_values", q_values)
            for i, (action, q_val) in enumerate(zip(action_names, q_values)):
                print(f"  {i}: {action:<20} Q-value: {q_val:.4f}")
            
            # Show best action
            best_action_idx = q_values.index(max(q_values))
            print(f"\nBest action: {action_names[best_action_idx]} (Q-value: {max(q_values):.4f})")
            
            await websocket.close()
            print("Connection closed")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting WebSocket test...")
    asyncio.run(test_robot_communication()) 