import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from datetime import datetime

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 30, 3, padding=1)  # Reduced from 32 to 30
        self.bn1 = nn.BatchNorm2d(30)
        self.dropout1 = nn.Dropout2d(0.10)
        self.conv2 = nn.Conv2d(30, 30, 3, padding=1)  # Reduced from 32 to 30
        self.bn2 = nn.BatchNorm2d(30)
        self.dropout2 = nn.Dropout2d(0.10)
        
        # 1x1 convolution to reduce channels after first block
        self.conv1x1_1 = nn.Conv2d(30, 16, 1)  # Reduced input from 32 to 30
        
        # Second Block
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout2d(0.10)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout2d(0.10)
        
        # 1x1 convolution to reduce channels after second block
        self.conv1x1_2 = nn.Conv2d(16, 8, 1)  # 7x7x8 (after maxpool)
        
        # Third Block
        self.conv5 = nn.Conv2d(8, 8, 3, padding=1)  # 7x7x8
        self.bn5 = nn.BatchNorm2d(8)
        self.dropout5 = nn.Dropout2d(0.10)
        self.conv6 = nn.Conv2d(8, 8, 3, padding=1)  # 7x7x8
        
        # Fully connected layer
        self.fc = nn.Linear(8 * 7 * 7, 10)  # Corrected input size
        
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
    
    # Create model and count parameters
    model = MNISTNet().to(device)
    print("\nModel Architecture:")
    print(model)
    total_params = count_parameters(model)
    print(f"\nModel size: {total_params/1e6:.2f}M parameters")
    
    # Enhanced data augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation((-10, 10)),  # Random rotation ±10 degrees
        transforms.RandomAffine(degrees=0, shear=(-5, 5)),  # Slight shear
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_accuracy = 0
    
    for epoch in range(20):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
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
        
        accuracy = 100. * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save model with timestamp and accuracy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), f'mnist_model_{timestamp}_{accuracy:.2f}.pth')
    
    print(f'Best Validation Accuracy: {best_accuracy:.2f}%')
    return model

if __name__ == "__main__":
    train_model() 