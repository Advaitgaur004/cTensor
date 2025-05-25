import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os
from datetime import datetime

class IrisModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_classes=3):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights using Glorot/Xavier initialization (similar to cTensor)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    print("=== pytorch results ===")
    print(f"time: {datetime.now()}")
    print("framework: pytorch")
    print(f"platform: {sys.platform}")
    print("")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"n_samples: {len(X)}")
    print(f"n_train_samples: {len(X_train)}")
    print(f"n_test_samples: {len(X_test)}")
    
    # Normalize the dataset (fit on training data only)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_norm)
    X_test_tensor = torch.FloatTensor(X_test_norm)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create model
    model = IrisModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    
    # Training parameters
    batch_size = 8
    num_epochs = 3
    
    # Train model
    model.train()
    for epoch in range(num_epochs):
        print(f"==> epoch: {epoch}")
        epoch_loss = 0.0
        num_batches = 0
        
        # Create batches
        for i in range(0, len(X_train_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_train_tensor))
            batch_X = X_train_tensor[i:batch_end]
            batch_y = y_train_tensor[i:batch_end]
            
            print(f" batch: {i}/{len(X_train_tensor)} samples")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        print(f"Epoch {epoch} average loss: {epoch_loss / num_batches:.6f}")
    
    # Evaluate model
    model.eval()
    correct = 0
    total = len(X_test_tensor)
    
    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            input_sample = X_test_tensor[i:i+1]  # Keep batch dimension
            true_label = y_test_tensor[i].item()
            
            # Forward pass
            output = model(input_sample)
            predicted = torch.argmax(output, dim=1).item()
            
            if predicted == true_label:
                correct += 1
            
            print(f"Sample {i} - True: {true_label}, Pred: {predicted}")
    
    accuracy = correct / total
    print(f"accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()