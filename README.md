# MNIST Classification with CI/CD Pipeline

This project implements a Deep Neural Network for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The implementation includes automated testing, model validation, and deployment processes.

## Model Architecture

The model (`MNISTNet`) is a Convolutional Neural Network with the following architecture:

```
MNISTNet(
  Input: 28x28 grayscale images
  (conv1): Conv2d(1, 16, kernel_size=3, padding=1)
  (bn1): BatchNorm2d(16)
  (dropout1): Dropout2d(p=0.25)
  MaxPool2d(2x2)
  
  (conv2): Conv2d(16, 32, kernel_size=3, padding=1)
  (bn2): BatchNorm2d(32)
  (dropout2): Dropout2d(p=0.25)
  MaxPool2d(2x2)
  
  (fc): Linear(32*7*7, 10)
  Output: 10 classes
)
```

### Key Features:
- Batch Normalization after each convolution layer
- Dropout (25%) for regularization
- MaxPooling for spatial dimension reduction
- Final Fully Connected layer for classification

## Training Process

- Dataset: MNIST
- Epochs: 20
- Optimizer: Adam with ReduceLROnPlateau scheduler
- Loss Function: Negative Log Likelihood
- Batch Size: 64
- Data Augmentation:
  * Random Rotation (±10 degrees)
  * Slight Shear (±5 degrees)
  * Normalization (mean=0.1307, std=0.3081)

## Testing Suite

The project includes comprehensive tests (`test_model.py`) that verify:

1. **Parameter Count Test**
   - Ensures model has less than 20,000 parameters
   - Validates model complexity constraints

2. **Input Shape Test**
   - Verifies model accepts 28x28 input images
   - Confirms output shape is correct (10 classes)

3. **Model Accuracy Test**
   - Tests model on MNIST test set
   - Ensures accuracy exceeds 99.4%
   - Prints detailed accuracy metrics

4. **Architecture Tests**
   - Validates presence of Batch Normalization
   - Confirms Dropout layers implementation
   - Verifies Fully Connected layer existence

## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automates:
1. Environment setup
2. Dependency installation
3. Model training
4. Test execution
5. Model artifact storage

## Project Structure

```
├── model.py           # Model architecture and training code
├── test_model.py      # Testing suite
├── requirements.txt   # Project dependencies
├── .gitignore        # Git ignore rules
└── .github/
    └── workflows/
        └── ml-pipeline.yml  # CI/CD configuration
```

## Local Development

1. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train model:

```bash
python model.py
```

4. Run tests:

```bash
pytest test_model.py -v

```

## Model Versioning

Models are saved with timestamps for versioning:
- Format: `mnist_model_YYYYMMDD_HHMMSS_accuracy.pth`
- Example: `mnist_model_20230615_143022_99.45.pth`

## GitHub Actions Integration

The pipeline automatically runs on every push:
1. Sets up Python 3.8 environment
2. Installs required packages
3. Trains the model
4. Runs all tests
5. Stores the trained model as an artifact

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest

## Notes

- The model is designed to be lightweight while maintaining >90% accuracy
- Training data is downloaded automatically when needed
- Model artifacts are preserved in GitHub Actions for each run

## Model Architecture Details

The model consists of three main blocks with 1x1 convolutions for channel reduction:

```
Block 1: Input (28x28) → Conv(32) → BN → ReLU → Dropout → Conv1x1(16) → MaxPool
Block 2: (14x14) → Conv(32) → BN → ReLU → Dropout → Conv1x1(16) → MaxPool
Block 3: (7x7) → Conv(32) → BN → ReLU → Dropout → Conv1x1(16)
Classification: Flatten → FC(16*7*7 → 10)
```

Key architectural features:
- Strategic use of 1x1 convolutions for channel reduction
- Consistent pattern of Conv → BN → ReLU → Dropout
- Total parameters: ~19,226 (under 20,000 limit)
- Progressive spatial reduction: 28x28 → 14x14 → 7x7