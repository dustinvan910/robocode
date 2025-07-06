import numpy as np


def action_adapter(action_values, action_prob=None, value=None, state_data=None):
    # print(f"Action: {action_values}, Action Prob: {action_prob}, Value: {value}, State Data: {state_data}")
   
    # If action_prob is provided, use it to select the action
    actionID = 0
    if action_prob is not None:
        # Convert action_prob to numpy array if it's a tensor
        if hasattr(action_prob, 'cpu'):
            action_prob = action_prob.cpu().numpy()
        
        # Handle log probabilities (negative values) from PPO agent
        if isinstance(action_prob, (list, np.ndarray)):
            # Convert log probabilities to regular probabilities
            if np.any(action_prob < 0):  # If we have negative values, they're log probabilities
                # print(f"Converting log probabilities: {action_prob}")
                action_prob = np.exp(action_prob)  # Convert log prob to prob
                # print(f"Converted to probabilities: {action_prob}")
            
            # If action_prob is a probability distribution, sample from it
            if len(action_prob) > 1:
                # Normalize probabilities to sum to 1
                action_prob = action_prob / np.sum(action_prob)
                actionID = np.random.choice(len(action_prob), p=action_prob)
                # print(f"Action ID: {actionID}", action_prob)
            else:
                # Single probability value
                actionID = int(action_values) if isinstance(action_values, (int, float)) else 0
        else:
            # Single value - check if it's a log probability
            if action_prob < 0:
                # print(f"Converting single log probability: {action_prob}")
                action_prob = np.exp(action_prob)  # Convert log prob to prob
                # print(f"Converted to probability: {action_prob}")
            
            # Use the original action
            actionID = int(action_values) if isinstance(action_values, (int, float)) else 0
            
    # print(f"Action ID: {actionID}")
    # print(f"Action: {action_values}")
    # print(f"Action Prob: {action_prob}")
    # print(f"Value: {value}")
    # print(f"State Data: {state_data}")
    
    # Convert NumPy types to Python native types for JSON serialization
    response = {
        "action": int(actionID)
    }
    
    # Add value if provided, ensuring it's a Python native type
    response["value"] = float(action_values[actionID])
    
    return response 