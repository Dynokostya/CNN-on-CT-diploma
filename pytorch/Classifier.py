import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns
import numpy as np


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, image_size):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def load_dataset(base_dir, image_size, split_percentage):
    all_image_paths = []
    all_labels = []
    inference_image_paths = []
    inference_labels = []

    for i, patient in enumerate(os.listdir(base_dir)[:14]):
        patient_path = os.path.join(base_dir, patient, 'DICOMCUT')
        patient_images = os.listdir(patient_path)
        if patient_images:
            for i in range(2):
                inference_image_paths.append(os.path.join(patient_path, patient_images[i]))
                inference_labels.append(patient)
            for img_file in patient_images[2:]:
                all_image_paths.append(os.path.join(patient_path, img_file))
                all_labels.append(patient)

    label_to_index = {label: idx for idx, label in enumerate(sorted(set(all_labels + inference_labels)))}
    label_indices = [label_to_index[label] for label in all_labels]
    inference_label_indices = [label_to_index[label] for label in inference_labels]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, label_indices, test_size=1 - split_percentage, random_state=42)

    train_ds = ImageDataset(train_paths, train_labels, image_size)
    val_ds = ImageDataset(val_paths, val_labels, image_size)
    inference_ds = ImageDataset(inference_image_paths, inference_label_indices, image_size)

    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(val_ds, batch_size=32,
                                                                         shuffle=False), DataLoader(inference_ds,
                                                                                                    batch_size=1,
                                                                                                    shuffle=False)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'---Epoch {epoch + 1}--- \nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n')

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, loader, criterion, device='cpu'):
    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100. * correct / total

    # Compute F1 Score, Precision, Recall
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(conf_matrix)

    return running_loss / len(loader), accuracy


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Передбачені класи')
    plt.ylabel('Реальні класи')
    plt.show()


def predict_inference(model, loader, device='cpu'):
    model = model.to(device)
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    print("Predicted:", predictions)
    print("Actual:   ", actuals)


def predict_inference_single(model, image, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = transform(image).unsqueeze(0)
    model = model.to(device)
    image = image.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    print("Predicted:", predicted.item())
    return predicted.item()


def plot_accuracy(train_accuracies, val_accuracies, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Точність на тренуванні, %')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Точність на валідації, %')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.legend()
    plt.show()


def plot_loss(train_losses, val_losses, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Втрата на тренуванні')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Втрата на валідації')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Set seed for reproducibility
    torch.manual_seed(12345)
    # Choose device based on availability of CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device:", torch.cuda.get_device_name(0))
    base_dir = 'PatientData'
    image_size = (512, 512)
    split_percentage = 0.8
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.006)
    train_loader, val_loader, inference_loader = load_dataset(base_dir, image_size, split_percentage)

    num_epochs = 5
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion,
                                                                             optimizer, num_epochs, device)
    plot_accuracy(train_accuracies, val_accuracies, num_epochs)
    plot_loss(train_losses, val_losses, num_epochs)

    # Evaluate model performance
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    # torch.save(model.state_dict(), 'Model5.pth')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    predict_inference(model, inference_loader, device)

