"""
PyTorch Multi-Layer Perceptron (MLP) for Tabular Classification.
Implements a modular neural network pipeline with Dataset, DataLoader,
loss functions, Adam optimizer, and training/validation loops.
"""

import sys

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TabularDataset(Dataset):
    """Custom PyTorch Dataset for tabular features and labels."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultilayerPerceptron(nn.Module):
    """Standard 3-layer Feedforward Neural Network with ReLU and Dropout."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp():
    if not TORCH_AVAILABLE:
        print("[Notice] PyTorch is not installed in the active environment.")
        print("Please install PyTorch using: pip install torch")
        return

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("Generating synthetic classification dataset (1000 samples, 20 features, 3 classes)...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_classes=3,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultilayerPerceptron(input_dim=20, hidden_dim=64, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 10
    print("\nStarting Training Loop:")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset) * 100

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    train_mlp()
