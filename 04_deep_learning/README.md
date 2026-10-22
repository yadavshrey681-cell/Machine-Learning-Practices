# Deep Learning & Neural Networks

Clean PyTorch implementations for neural network architectures, custom datasets, and modular training/evaluation loops.

## Modules
- **`pytorch_mlp.py`**: Multi-Layer Perceptron (MLP) for tabular classification with standard `torch.nn.Module`, `Dataset`, `DataLoader`, optimizer, and validation tracking.
- **`pytorch_cnn.py`**: Convolutional Neural Network (CNN) template for image classification featuring convolutional layers, max pooling, dropout regularization, and cross-entropy loss.

## How to Run
```bash
python 04_deep_learning/pytorch_mlp.py
python 04_deep_learning/pytorch_cnn.py
```
*(Ensure `torch` is installed via `pip install torch`)*
