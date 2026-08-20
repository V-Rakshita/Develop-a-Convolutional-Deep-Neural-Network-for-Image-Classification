# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Load the Fashion-MNIST dataset and prepare the images by converting them into tensors and normalizing the pixel values.

### STEP 2:

Create DataLoaders to divide the training and testing data into small batches for easier and faster processing.

### STEP 3:

Design a CNN model with three convolutional layers and max-pooling layers to extract important features from the images.

### STEP 4:

Add fully connected layers to classify the extracted features into one of the 10 Fashion-MNIST clothing categories.

### STEP 5:

Train the CNN using the Adam optimizer and Cross-Entropy Loss for multiple epochs so that the model learns from the training images.

### STEP 6:

Test the trained model using the test dataset and evaluate its performance using accuracy, confusion matrix, classification report, and sample image prediction.





## PROGRAM

### Name:

### Register Number:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

im, lbl = train_set[0]

print("Image Shape:", im.shape)
print("Training Dataset Size:", len(train_set))
print("Testing Dataset Size:", len(test_set))

trl = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

tstl = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)

class CNNclassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.c2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.c3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.l1 = nn.Linear(
            128 * 3 * 3,
            64
        )

        self.l2 = nn.Linear(
            64,
            32
        )

        self.l3 = nn.Linear(
            32,
            10
        )

    def forward(self, x):

        x = self.pool(
            torch.relu(
                self.c1(x)
            )
        )

        x = self.pool(
            torch.relu(
                self.c2(x)
            )
        )

        x = self.pool(
            torch.relu(
                self.c3(x)
            )
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = torch.relu(
            self.l1(x)
        )

        x = torch.relu(
            self.l2(x)
        )

        x = self.l3(x)

        return x

model = CNNclassifier()

if torch.cuda.is_available():
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda")
else:
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cpu")

model.to(device)

summary(
    model,
    input_size=(1, 28, 28)
)

criterion = nn.CrossEntropyLoss()

op = optim.Adam(
    model.parameters(),
    lr=0.0005
)

epochs = 3

for i in range(epochs):

    model.train()

    rl = 0.0

    for im, lbl in trl:

        im = im.to(device)
        lbl = lbl.to(device)

        op.zero_grad()

        pred = model(im)

        loss = criterion(
            pred,
            lbl
        )

        loss.backward()

        op.step()

        rl += loss.item()

    print(
        f"Running Loss: {i+1}/{epochs}",
        rl / len(trl)
    )

act = []
pre = []

crt = 0
total = 0

model.eval()

with torch.no_grad():

    for im, lbl in tstl:

        im = im.to(device)
        lbl = lbl.to(device)

        output = model(im)

        _, pred = torch.max(
            output,
            1
        )

        total += lbl.size(0)

        crt += (
            pred == lbl
        ).sum().item()

        pre.extend(
            pred.cpu().numpy()
        )

        act.extend(
            lbl.cpu().numpy()
        )

accuracy = (crt / total) * 100

print("Accuracy:", accuracy)

accuracy_sklearn = accuracy_score(
    act,
    pre
) * 100

print(
    "Accuracy using accuracy_score:",
    accuracy_sklearn
)

conf_matrix = confusion_matrix(
    act,
    pre
)

print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(
    act,
    pre,
    target_names=test_set.classes
)

print("Classification Report:")
print(class_report)

import seaborn as sns

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=test_set.classes,
    yticklabels=test_set.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

with torch.no_grad():

    img1, label = test_set[0]

    img1_input = img1.unsqueeze(0)

    img1_input = img1_input.to(device)

    output = model(img1_input)

    _, pred = torch.max(
        output,
        1
    )

    classes = test_set.classes

    img1 = img1 * 0.5 + 0.5

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(
        img1.squeeze(),
        cmap="gray"
    )

    plt.title(
        "Predicted Image"
    )

    plt.axis('off')

    plt.show()

    print(
        f"Actual: {classes[label]}"
    )

    print(
        f"Predicted: {classes[pred.item()]}"
    )

    print("\nRaw Model Output:")
    print(output.cpu())

    print(
        "\nPredicted Class Index:",
        pred.item()
    )

    print(
        "Actual Class Index:",
        label
    )

```

### OUTPUT

## Training Loss per Epoch

<img width="415" height="78" alt="image" src="https://github.com/user-attachments/assets/349ef42f-7d80-48da-8d62-fc467d542e06" />


## Confusion Matrix

<img width="437" height="127" alt="image" src="https://github.com/user-attachments/assets/08777cbd-dda8-4498-aa60-d3fffbc9e741" />

<img width="685" height="611" alt="image" src="https://github.com/user-attachments/assets/e1585fc1-cfae-44ca-b869-f78ddaa01c57" />



## Classification Report
<img width="553" height="407" alt="image" src="https://github.com/user-attachments/assets/49c089a7-52d6-460d-aafe-630d2e275924" />


### New Sample Data Prediction
<img width="776" height="651" alt="image" src="https://github.com/user-attachments/assets/b1a995fd-8f01-4881-847f-2bc32a3abdcf" />


## RESULT
Therefore, a convolutional deep neural network (CNN) for image classification and to verify the response for new images has been developed.
