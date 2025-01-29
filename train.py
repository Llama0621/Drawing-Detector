import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split

class QuickDrawDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert array to PIL Image
        image = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class EnhancedQuickDrawNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

def download_quickdraw_data(categories, samples_per_category=2000):
    images = []
    labels = []
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    
    os.makedirs('data', exist_ok=True)
    
    for idx, category in enumerate(categories):
        print(f"Processing {category}...")
        
        file_name = f"{category}.npy"
        file_path = os.path.join('data', file_name)
        url = f"{base_url}{category}.npy"
        
        if not os.path.exists(file_path):
            print(f"Downloading {category} dataset...")
            try:
                download_file(url, file_path)
            except Exception as e:
                print(f"Error downloading {category}: {str(e)}")
                continue
        
        try:
            category_data = np.load(file_path)
            print(f"Found {len(category_data)} samples for {category}")
            
            selected_data = category_data[:samples_per_category]
            selected_data = selected_data.reshape(-1, 28, 28)
            
            images.extend(selected_data)
            labels.extend([idx] * len(selected_data))
            
            print(f"Successfully processed {len(selected_data)} samples for {category}")
            
        except Exception as e:
            print(f"Error processing {category}: {str(e)}")
            continue
    
    if not images:
        raise ValueError("No images were collected. Please check the category names and try again.")
    
    return np.array(images), np.array(labels)

def train_model(model, train_loader, val_loader, device, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Mixup augmentation
            if np.random.random() > 0.5:
                lam = np.random.beta(0.5, 0.5)
                rand_index = torch.randperm(images.size()[0]).to(device)
                mixed_x = lam * images + (1 - lam) * images[rand_index]
                optimizer.zero_grad()
                outputs = model(mixed_x)
                loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[rand_index])
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)
        
        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_quickdraw_model.pth')
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print("Early stopping triggered")
            break
        
        print('-' * 50)
    
    return best_val_acc

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    categories = ['cat', 'dog', 'fish', 'bird', 'house']
    samples_per_category = 2000
    batch_size = 64
    
    try:
        print("Downloading QuickDraw data...")
        images, labels = download_quickdraw_data(categories, samples_per_category)
        
        print(f"Total images collected: {len(images)}")
        print(f"Total labels collected: {len(labels)}")
        print(f"Image shape: {images[0].shape}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = QuickDrawDataset(X_train, y_train, transform=get_transforms(train=True))
        val_dataset = QuickDrawDataset(X_val, y_val, transform=get_transforms(train=False))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = EnhancedQuickDrawNet(num_classes=len(categories)).to(device)
        
        best_acc = train_model(model, train_loader, val_loader, device)
        print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()