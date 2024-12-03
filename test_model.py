import torch
import torch.nn as nn
import torchvision
from model import MNISTNet
import pytest

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = MNISTNet()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, should be less than 20000"

def test_input_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet().to(device)
    
    try:
        import glob
        model_files = glob.glob('mnist_model_*.pth')
        if not model_files:
            from model import train_model
            model = train_model()
        else:
            latest_model = max(model_files, key=lambda x: x.split('_')[1])
            try:
                model.load_state_dict(torch.load(latest_model))
                print(f"Loaded model: {latest_model}")
            except:
                print("Error loading existing model, training new one...")
                from model import train_model
                model = train_model()
    except Exception as e:
        pytest.fail(f"Failed to load or train model: {str(e)}")
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
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
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    assert accuracy > 99.4, f"Model accuracy is {accuracy:.2f}%, should be > 99.4%"

def test_architecture():
    model = MNISTNet()
    
    # Check for BatchNorm layers
    has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_batchnorm, "Model should use BatchNormalization"
    
    # Check for Dropout layers
    has_dropout = any(isinstance(m, nn.Dropout2d) for m in model.modules())
    assert has_dropout, "Model should use Dropout"
    
    # Check for Fully Connected layer
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    assert has_fc, "Model should have a Fully Connected layer"

if __name__ == "__main__":
    pytest.main([__file__]) 