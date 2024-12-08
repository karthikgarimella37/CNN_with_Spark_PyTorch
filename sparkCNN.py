# Libraries
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.optim import lr_scheduler
import os
import pandas as pd
import numpy as np
import argparse
from torch.optim.lr_scheduler import StepLR

# Import Spark libraries
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
# Set up Spark session and context
conf = SparkConf().setAppName("DistributedCNN").setMaster("local[*]")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext

# GPU if available
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Data Augmentation for test and train
transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=np.array([0.5, 0.5, 0.5]),
                             std=np.array([0.25, 0.25, 0.25]))
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=np.array([0.5, 0.5, 0.5]),
                             std=np.array([0.25, 0.25, 0.25]))
    ]),

}

# Training and Testing Data
data_dir = "/home/sat3812/Downloads/xray_dataset_covid19"

train_data = ImageFolder(os.path.join(data_dir, "train"), transform = transform["train"])
test_data = ImageFolder(os.path.join(data_dir, "test"), transform = transform["test"])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

# Validate on the test set
def validate(model, test_loader, criterion, device = device):
  model.eval()
  totalValLoss = 0.0
  valPredictions = 0
  totalSamples = 0
  all_labels = []
  all_predictions = []
  all_probs = []

  with torch.no_grad():
    for data, labels in test_loader:
      valInputs = data.to(device)
      valLabels = labels.to(device)
      valOutputs = model(valInputs)
      valLoss = criterion(valOutputs, valLabels)

      totalValLoss += valLoss.item()

      _, predictedVal = torch.max(valOutputs.data, 1)
      probabilities = torch.softmax(valOutputs, dim=1)[:, 1]
      valPredictions += (predictedVal == valLabels).sum().item()

      all_labels.extend(valLabels.cpu().numpy())
      all_predictions.extend(predictedVal.cpu().numpy())
      all_probs.extend(probabilities.cpu().numpy())
      
      totalSamples += valLabels.size(0)
  valAccuracy = (valPredictions / totalSamples) * 100
  valLoss = totalValLoss / len(test_loader)
  cm = confusion_matrix(all_labels, all_predictions)
  TN, FP, FN, TP = cm.ravel()

  accuracy = (TP + TN) / (TP + TN + FP + FN)
  sensitivity = recall_score(all_labels, all_predictions)  # Same as recall
  specificity = TN / (TN + FP)
  precision = precision_score(all_labels, all_predictions)
  f1 = f1_score(all_labels, all_predictions)

  auc_score = roc_auc_score(all_labels, all_probs)
  fpr, tpr, _ = roc_curve(all_labels, all_probs)

  return valAccuracy, valLoss, sensitivity, f1, cm, auc_score, fpr, tpr



# Max-Pooling function
def maxpool_output_shape(inputHeight, inputWidth, poolSize = 2):
  outputHeight = int(inputHeight/poolSize)
  outputWidth = int(inputWidth/poolSize)
  return outputHeight, outputWidth

# 2D Convolutional parameters
def findConv2d_shape(height, width, conv):
  kernel_size = conv.kernel_size
  stride = conv.stride
  padding = conv.padding
  dilation = conv.dilation

  # Calculate output height and width
  height = np.floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
  width = np.floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

  return int(height), int(width)

# Train function
def train(model, train_loader, optimizer, criterion, scheduler = None,
          test_loader = None, epochs = 10, device = device):
  for epoch in range(epochs):
    model.train()
    loss = 0.0
    trainPredictions = 0
    totalSamples = 0
    for data, labels in train_loader:
      inputs = data.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()

      outputs = model(inputs)
      trainLoss = criterion(outputs, labels)
      trainLoss.backward()
      optimizer.step()

      if scheduler:
        scheduler.step()

      loss += trainLoss.item()

      _, predicted = torch.max(outputs.data, 1)
      trainPredictions += (predicted == labels).sum().item()
      totalSamples += labels.size(0)
    trainAccuracy = trainPredictions / totalSamples

    print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {loss / len(train_loader):.3f}, Training Accuracy: {trainAccuracy * 100:.3f}%')
    if test_loader is not None:
      
      val_accuracy, val_loss, recall, f1, cm, auc_score, fpr, tpr = validate(model, test_loader, criterion,device=device)
#      print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}%\n')
      print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}%, Recall: {recall:.3f}, F1: {f1:.3f}')
      print(f'Confusion Matrix:\n{cm}')


    print('Training finished')
  return cm, auc_score, fpr, tpr, train_loss_lst, val_loss_lst

    # End of train function

config = {
    "input_shape": (3, 224, 224),
    "classes": 2
}


# Seed for reproduceability
randomSeed = 37
torch.manual_seed(randomSeed)

# Class CNN for CNN architecture
class CNN(torch.nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        conv, height, width = config["input_shape"]
        classes = config["classes"]

        # Block 1: Conv -> ReLU -> BatchNorm -> Conv -> ReLU -> BatchNorm -> MaxPool
        self.conv1 = torch.nn.Conv2d(conv, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        height, width = maxpool_output_shape(*findConv2d_shape(*findConv2d_shape(height, width, self.conv1), self.conv2), 2)

        # Block 2: Conv -> ReLU -> BatchNorm -> Conv -> ReLU -> BatchNorm -> MaxPool
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm2d(256)

        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        height, width = maxpool_output_shape(*findConv2d_shape(*findConv2d_shape(height, width, self.conv3), self.conv4), 2)

        # Block 3: Conv -> ReLU -> BatchNorm -> Dropout -> MaxPool
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu5 = torch.nn.ReLU()
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.dropout1 = torch.nn.Dropout(0.5)

        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        height, width = maxpool_output_shape(*findConv2d_shape(height, width, self.conv5), 2)

        # Fully connected layers
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512 * height * width, 512)
        self.relu_fc1 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.5)

        self.fc2 = torch.nn.Linear(512, 128)
        self.relu_fc2 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(128, classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.maxpool1(x)

        # Block 2
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        x = self.maxpool2(x)

        # Block 3
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.dropout1(x)

        x = self.maxpool3(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)

        x = self.fc3(x)
        return x

# Loss function, Adam optimizer and Scheduler
model = CNN(config).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)



print(model)

# Training the model
train(model, train_loader, optimizer, criterion,
          scheduler=scheduler, test_loader=test_loader, epochs=50, device=device)



# ResNet18 Model

resnet18_model = models.resnet18(weights=True)

num_ftrs = resnet18_model.fc.in_features
resnet18_model.fc = torch.nn.Linear(num_ftrs,2)
resnet18_model.to(device)
criterion_resnet = torch.nn.CrossEntropyLoss()

# SGD used for optimization
optimizer_resnet = torch.optim.SGD(resnet18_model.parameters(), lr=0.001)

# Training the Resnet18 Model
train(resnet18_model, train_loader, optimizer_resnet, criterion_resnet, scheduler=None,
          test_loader=test_loader, epochs=10, device=device)


# Stop the Spark session
spark.stop()

