# CNN DQN for Robocode (Grayscale Image Input)

This implementation provides a Convolutional Neural Network (CNN) Deep Q-Network (DQN) agent for Robocode that processes grayscale battlefield images. The CNN DQN can analyze visual information from the battlefield, making it potentially more effective at understanding spatial relationships and enemy positions while using less computational resources than RGB processing.

## Features

- **Grayscale Image Processing**: Uses CNN layers to process battlefield images (64x64 grayscale)
- **Deep CNN Architecture**: 4 convolutional layers with pooling for feature extraction
- **Experience Replay**: Stores and samples experiences for stable training
- **Target Network**: Uses a separate target network for more stable Q-value estimates
- **Gradient Clipping**: Prevents exploding gradients during training
- **Dropout Regularization**: Reduces overfitting
- **Efficient Processing**: Grayscale reduces input complexity and memory usage

## Architecture

### CNN Network Structure
- **Input**: 64x64 grayscale battlefield image (1 channel)
- **Conv1**: 32 filters, 3x3 kernel, ReLU activation
- **Conv2**: 64 filters, 3x3 kernel, ReLU activation + MaxPool
- **Conv3**: 128 filters, 3x3 kernel, ReLU activation + MaxPool
- **Conv4**: 128 filters, 3x3 kernel, ReLU activation + MaxPool
- **FC1**: 512 units, ReLU + Dropout
- **FC2**: 512 units, ReLU + Dropout
- **Output**: 8 action values

### Image Processing
The agent processes battlefield images with the following steps:

1. **Base64 Decoding**: Decodes base64-encoded image from state data
2. **Grayscale Conversion**: Converts to grayscale format (L mode)
3. **Resizing**: Resizes to 64x64 pixels for consistent input
4. **Normalization**: Normalizes pixel values to [0, 1] range
5. **Channel Format**: Converts to PyTorch format (1, H, W)

## Files

- `cnn_dqn_agent_pytorch.py`: CNN DQN agent implementation for grayscale image input
- `main_cnn_pytorch.py`: Main training script for grayscale image processing
- `train_cnn_dqn.py`: Training script with command-line options
- `requirements_cnn.txt`: Python dependencies including image processing
- `README_CNN_DQN.md`: This documentation

## Installation

1. Install dependencies:
```bash
pip install -r requirements_cnn.txt
```

2. Make sure your Robocode robot is configured to send battlefield images via WebSocket.

## Usage

### Basic Training
```bash
python main_cnn_pytorch.py
```

### Training with Custom Parameters
```bash
python train_cnn_dqn.py --episodes 2000 --learning-rate 0.0001 --epsilon 0.8
```

### Play Mode (No Learning)
```bash
python train_cnn_dqn.py --play
```

### Command Line Options
- `--episodes`: Number of training episodes (default: 1000)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon-decay`: Epsilon decay rate (default: 0.999999)
- `--batch-size`: Batch size for training (default: 32)
- `--memory-size`: Replay memory size (default: 50000)
- `--play`: Run in play mode without learning
- `--port`: WebSocket port (default: 5000)

## Model Files

The agent automatically saves and loads model files:
- `cnn_dqn_model_pytorch.pth`: Trained model weights
- `cnn_grayscale_training_stats_pytorch.json`: Training statistics

## State Data Format

The agent expects state data in JSON format with an image field:

```json
{
  "image": "base64_encoded_image_data",
  "reward": 0.5,
  "time": 100,
  "isWin": false
}
```

The image should be a battlefield screenshot encoded in base64 format (will be converted to grayscale automatically).

## Advantages over Regular DQN

1. **Visual Understanding**: Can directly process battlefield images
2. **Spatial Awareness**: Understands battlefield layout and positions
3. **Translation Invariance**: CNN layers can recognize patterns regardless of exact position
4. **Hierarchical Features**: Learns low-level features (edges, corners) to high-level features (enemy positions, safe zones)
5. **Better Generalization**: More robust to variations in battlefield conditions
6. **Efficient Processing**: Grayscale reduces computational complexity compared to RGB

## Advantages of Grayscale over RGB

1. **Reduced Complexity**: 1 channel vs 3 channels reduces input size by 66%
2. **Faster Training**: Fewer parameters to learn
3. **Lower Memory Usage**: Reduced memory footprint
4. **Better for Battlefield**: Grayscale is often sufficient for detecting robots, walls, and obstacles
5. **Less Overfitting**: Smaller input space reduces overfitting risk

## Training Tips

1. **Start with Exploration**: Use high epsilon (0.8-1.0) initially
2. **Monitor Loss**: Watch for stable or decreasing loss values
3. **Check Statistics**: Review training stats to ensure learning
4. **Save Regularly**: Models are saved every 10 episodes
5. **Use GPU**: If available, the agent will automatically use CUDA
6. **Image Quality**: Ensure battlefield images are clear and consistent
7. **Grayscale Optimization**: Grayscale processing is faster and more memory efficient

## Performance Monitoring

The agent tracks various metrics:
- Episode rewards
- Win/loss rates
- Training loss
- Epsilon decay
- Optimal action distribution

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **No Learning**: Check epsilon decay and learning rate
3. **Connection Issues**: Verify WebSocket port and robot configuration
4. **Poor Performance**: Try adjusting learning rate or network architecture
5. **Image Processing Errors**: Check image format and base64 encoding
6. **Grayscale Conversion**: Ensure images are properly converted to grayscale

## Comparison with Regular DQN

| Feature | Regular DQN | CNN DQN (Grayscale) |
|---------|-------------|---------------------|
| Input Type | Numerical State | Grayscale Battlefield Images |
| State Processing | Fully Connected | CNN + Fully Connected |
| Spatial Awareness | Limited | High |
| Parameter Count | Lower | Higher |
| Training Time | Faster | Slower |
| Memory Usage | Lower | Higher |
| Visual Understanding | None | High |
| Input Channels | N/A | 1 (Grayscale) |

## Comparison with RGB CNN DQN

| Feature | RGB CNN DQN | Grayscale CNN DQN |
|---------|-------------|-------------------|
| Input Channels | 3 | 1 |
| Memory Usage | Higher | Lower |
| Training Speed | Slower | Faster |
| Computational Cost | Higher | Lower |
| Feature Richness | Higher | Lower |
| Suitability for Battlefield | Good | Excellent |

## Future Improvements

1. **Attention Mechanisms**: Add attention layers for better focus on important image regions
2. **Multi-Scale Processing**: Process different resolution images
3. **Recurrent Layers**: Add LSTM/GRU for temporal dependencies
4. **Auxiliary Tasks**: Predict enemy movement, energy changes from images
5. **Ensemble Methods**: Combine multiple CNN models
6. **Data Augmentation**: Rotate, flip, or adjust brightness of training images
7. **Transfer Learning**: Use pre-trained CNN models for feature extraction
8. **Hybrid Input**: Combine grayscale images with numerical state data 