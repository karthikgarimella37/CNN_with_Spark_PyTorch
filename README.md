# CNN using Spark and PyTorch

Using Spark and PyTorch, we perform classification of Pneumonia Chest X-ray among COVID-19 patients using CNN.

## Problems intended to be solved:
+ The main idea of this project is to leverage CNN to recognise specific patterns in Pneumonia patients.
+ The 2019 coronavirus (COVID-19) presents several unique features for Pneumonia patients. CNN’s image recognition capabilities can be used on the chest X-rays of patients to determine if the patient has Pneumonia.
+ A deep learning model is developed to identify pneumonia patterns on X-rays of COVID-19 patients. Spark will be used for big image data analysis in a distributed environment.


## Data

+ The data is taken from: [Kaggle Covid19 Xray Dataset](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/data)

+ The dataset was created by Joseph Paul Cohen and Paul Morrison and Lan Dao. COVID-19 image data collection, arXiv, 2020. https://github.com/ieee8023/covid-chestxray-dataset.  

![Dataset Image] (/Images/Dataset_Image.png)


## Data Split
The data is split into Train and Test by default, in an approximate 80-20 split. The response variable is perfectly balanced.

![Class Distribution](/Images/Class_Distribution.png)


Sample size: 188 images
Train Frequency: 74 True and 74 False
Test Frequency: 20 True and 20 False

## Data Augmentation

+ Resizing some of the images
+ CenterCrop images to create sets focused on the center of the image
+ Normalized the images to improve pixel density.

![Augmented Dataset Image](/Images/Augmented_Images.png)

# Custom CNN Model

## Custom Model Architecture
A custom CNN model with 5 Convolutional Layers, ReLU as the activation function with Batch Normalization, MaxPooling, Dropout Layers and 3 Fully Connected Layers was created.

![Custom CNN Architecture](/Images/Custom_CNN_Architecture.png)

## Model Hyperparameters

+ Criterion: CrossEntropyLoss
+ Batch Size: 16
+ Optimizer: Adam
+ Learning Rate: 0.0001
+ Epochs: 50 
+ Scheduler: StepLR with Stepsize = 1, Gamma = 0.7

## Model Performance

After 50 epochs:
Epoch [50/50], Training Loss: 0.833, Training Accuracy: 83.108%
Epoch [50/50], Validation Loss: 0.505, Validation Accuracy: 75.0%, Recall: 0.900, F1: 0.783

Confusion Matrix:

![Custom CNN CM](/Images/Custom_Model_CM.png)

ROC Curve (with AUC score):

![Custom ROC](/Images/Custom_Model_ROC.png)

Model Performance:

![Custom Model Performance](/Images/Custom_Model_Performance.png)


# Pretrained Model (ResNet18)

Resnet18 Model Architecture:

![Resnet18 Arch](/Images/ResNet18_Architecture.png)

ResNet18 Model Performance:

After 10 epochs:
Epoch [10/10], Training Loss: 0.082, Training Accuracy: 99.324%
Epoch [10/10], Validation Loss: 0.176, Validation Accuracy: 92.5%, Recall: 0.900, F1: 0.923

Confusion Matrix:

![Resnet18 CM](/Images/ResNet18_CM.png)

ROC Curve (with AUC score):

![Resnet18 ROC](/Images/ResNet18_ROC.png)

Model Performance:

![Resnet18 Model Performance](/Images/ResNet18_Model_Performance.png)









