# DFedRPT

A comprehensive research platform for experimenting with various federated learning algorithms and personalization techniques.

## Overview

DFedRPT implements multiple state-of-the-art federated learning algorithms including FedAvg, DisPFL, ProxyFL, Ditto, and others. The platform supports various datasets and model architectures for comprehensive evaluation of federated learning approaches.

## Supported Algorithms

- **FedAvg**: Standard Federated Averaging
- **DisPFL**: Distributed Personalized Federated Learning  
- **ProxyFL**: Proxy-based Federated Learning
- **Ditto**: Federated Learning with Personalization
- **DFedRPT**: Federated Learning with Proximal Regularization
- **DFedAvg**: Dynamic Federated Averaging
- **AvgPush**: Average Push Algorithm

## Supported Datasets

- MNIST / Fashion-MNIST
- CIFAR-10 / CIFAR-100
- CIFAR-10N (Noisy)
- ISIC2018 (Medical Imaging)
- HAR (Human Activity Recognition)

## Model Architectures

- **CNN**: Convolutional Neural Network
- **ResNet18**: Residual Network
- **HARCNN**: Specialized CNN for HAR dataset
- **Proxy Models**: MLR (Multinomial Logistic Regression) and CNN variants

## Usage

### Basic Training

Run federated learning with default settings:
```bash
python main.py
```

### Custom Configuration

Train with specific algorithm and dataset:
```bash
python main.py --algorithm FedAvg --dataset cifar10 --model_str cnn --num_clients 10 --global_rounds 100
```

### Advanced Options

```bash
python main.py \
    --algorithm ProxyFL \
    --dataset fmnist \
    --model_str resnet \
    --proxy_model mlr \
    --num_clients 20 \
    --global_rounds 150 \
    --local_epochs 5 \
    --local_learning_rate 0.01 \
    --join_ratio 0.8
```

### Batch Execution

Use the provided shell script for batch experiments:
```bash
bash run.sh
```

## Key Parameters

### General Settings
- `--algorithm`: FL algorithm to use (FedAvg, DisPFL, ProxyFL, etc.)
- `--dataset`: Dataset name (fmnist, cifar10, cifar100, etc.)
- `--model_str`: Model architecture (cnn, resnet, harcnn)
- `--num_clients`: Total number of clients
- `--global_rounds`: Number of global training rounds
- `--local_epochs`: Local training epochs per round

### Learning Parameters
- `--local_learning_rate`: Client learning rate
- `--batch_size`: Local batch size
- `--join_ratio`: Fraction of clients participating per round
- `--optimizer`: Optimizer type (SGD, Adam)


## Project Structure

```
DFedRPT/
├── core/
│   ├── flimplement/
│   │   ├── clients/           # Client implementations
│   │   ├── middleware/        # Server/algorithm implementations
│   │   └── model/            # Neural network models
│   └── utils/                # Utility functions
├── data/                     # Data generation scripts
├── main.py                   # Main execution script
└── run.sh                    # Batch execution script
```

## Data Generation

Generate federated datasets:
```bash
# Generate CIFAR-10 with non-IID distribution
python data/generate_cifar10.py

# Generate Fashion-MNIST
python data/generate_fmnist.py

# Generate ISIC2018 medical dataset
python data/generate_ISIC.py
```

## Results and Logging

Results are automatically saved in the `results/` directory, organized by dataset. Each run generates:
- Training logs with detailed metrics
- Test accuracy, precision, recall, and F1 scores
- Training loss curves

## Research Applications

This platform supports research in:
- Federated Learning algorithms
- Personalization techniques
- Non-IID data handling
- Communication efficiency
- Privacy-preserving machine learning


