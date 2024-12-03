import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 28, 3, padding=1)  # Reduced from 30 to 28
        self.bn1 = nn.BatchNorm2d(28)
        self.dropout1 = nn.Dropout2d(0.10)
        self.conv2 = nn.Conv2d(28, 28, 3, padding=1)  # Reduced from 30 to 28
        self.bn2 = nn.BatchNorm2d(28)
        self.dropout2 = nn.Dropout2d(0.10)
        
        # 1x1 convolution to reduce channels after first block
        self.conv1x1_1 = nn.Conv2d(28, 14, 1)  # Reduced from 16 to 14
        
        # Second Block
        self.conv3 = nn.Conv2d(14, 14, 3, padding=1)  # Reduced from 16 to 14
        self.bn3 = nn.BatchNorm2d(14)
        self.dropout3 = nn.Dropout2d(0.10)
        self.conv4 = nn.Conv2d(14, 14, 3, padding=1)  # Reduced from 16 to 14
        self.bn4 = nn.BatchNorm2d(14)
        self.dropout4 = nn.Dropout2d(0.10)
        
        # 1x1 convolution to reduce channels after second block
        self.conv1x1_2 = nn.Conv2d(14, 8, 1)  # Keep at 8
        
        # Third Block
        self.conv5 = nn.Conv2d(8, 8, 3, padding=1)  # Keep at 8
        self.bn5 = nn.BatchNorm2d(8)
        self.dropout5 = nn.Dropout2d(0.10)
        self.conv6 = nn.Conv2d(8, 8, 3, padding=1)  # Keep at 8
        
        # Fully connected layer
        self.fc = nn.Linear(8 * 7 * 7, 10)
        
    def forward(self, x):
        # First Block
        x = self.conv1(x)          # 28x28x32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)          # 28x28x32
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = F.max_pool2d(x, 2)     # 14x14x32
        x = self.conv1x1_1(x)      # 14x14x16
        
        # Second Block
        x = self.conv3(x)          # 14x14x16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.conv4(x)          # 14x14x16
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = F.max_pool2d(x, 2)     # 7x7x16
        x = self.conv1x1_2(x)      # 7x7x8
        
        # Third Block
        x = self.conv5(x)          # 7x7x8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.conv6(x)          # 7x7x8
        
        # Classification
        x = x.view(-1, 8 * 7 * 7)  # Flatten
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    """Count and print trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print parameter count for each layer
    print("\nParameter count by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    return total_params

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced data augmentation with more subtle transformations
    transform_train = transforms.Compose([
        transforms.RandomRotation((-8, 8)),  # Reduced rotation range
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.08, 0.08),  # Add small translations
            scale=(0.95, 1.05),      # Add slight scaling
            shear=(-5, 5)
        ),
        transforms.ColorJitter(brightness=0.2),  # Added slight brightness variation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Add validation set
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)  # Increased initial learning rate
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.2,      # More aggressive LR reduction
        patience=1,      # Reduced patience for faster adaptation
        verbose=True
    )
    
    best_accuracy = 0
    patience_counter = 0
    max_patience = 4    # Adjusted early stopping patience for 20 epochs
    
    for epoch in range(20):  # Keep at 20 epochs
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = round(100. * correct / total, 1)  # Round to 1 decimal
        print(f'Validation Accuracy: {accuracy:.1f}%')
        
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            # Save model with timestamp and accuracy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), f'mnist_model_{timestamp}_{accuracy:.1f}.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break
    
    print(f'Best Validation Accuracy: {best_accuracy:.2f}%')
    return model

if __name__ == "__main__":
    train_model() 