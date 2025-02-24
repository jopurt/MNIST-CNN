import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageOps
import cv2

import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



from torchvision.transforms import v2

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")
# device = "cuda"

def convert_to_bnw(file_name):
    img = cv2.imread(file_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, img_bnw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)

    img_converted = Image.fromarray(img_bnw)
    img_converted = ImageOps.invert(img_converted)

    # img_converted.show()

    return img_converted


def convert_to_bnw2(file_name):
    img_gray = np.array(Image.open(file_name).convert('L'))

    thresh = 93
    maxval = 255

    im_bin = (img_gray > thresh) * maxval

    im_bin = Image.fromarray(np.uint8(im_bin))

    im_bin = ImageOps.invert(im_bin)

    # im_bin.show()

    return im_bin

def ensure_binary_tensor(image_tensor):

    return torch.where(image_tensor > 0.05, torch.tensor(1.0), torch.tensor(0.0))

transform_img = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: ensure_binary_tensor(x)),
    transforms.Normalize((0.5,), (0.5,))

])

image_paths = ['000.png', '333.png', '777.png']

# for i in range(len(image_paths)):
#     image = mpimg.imread(image_paths[i])
#     plt.imshow(image)
#     plt.show()

images = []
for path in image_paths:
    img = convert_to_bnw2(path)
    # img = Image.open(path)

    img_tensor = transform_img(img)
    # img_tensor = binarize_tensor(img_tensor)
    images.append(img_tensor)

for i in range(len(images)):
    # print(images[i].shape)
    image = images[i].squeeze().numpy()
    # print(image.shape)
    plt.imshow(image, cmap="gray")
    plt.axis()
    plt.show()

# exit(0)

images_tensor = torch.stack(images)
images_labels = torch.tensor([0, 3, 7])

class CustomTestDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

mydataset = CustomTestDataset(images_tensor, images_labels)
mytest_loader = DataLoader(mydataset,shuffle=False)

device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.reshape(-1, 32 * 14 * 14)
        x = self.fc1(x)
        return x


model = CNN().to(device=device)
model.load_state_dict(torch.load('cnn_model_weights.pth'))

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer.load_state_dict(torch.load('cnn_optimizer_state.pth'))


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(best_loss, epoch):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), 'cnn_model_weights.pth')
        torch.save(optimizer.state_dict(), 'cnn_optimizer_state.pth')

    if epoch % 10 == 0:
        print(f"Best loss: {best_loss}")

    return best_loss

epochs = 100
best_loss = 1000.

# for i in range(epochs):
#     print('epoch',i)
#     best_loss = train(best_loss, i)
#
# print("Done!")


# for epoch in range(num_epochs):
#     model.train()
#     # running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(images)
#
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # running_loss += loss.item()
#
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# model.eval()
correct = 0
total = 0

# with torch.no_grad():
#     for images, labels in train_loader:
#         print(labels)
#         images, labels = images.to(device), labels.to(device)
#         # print(images.shape)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f"Accuracy: {100 * correct / total:.2f}%")


y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

correct = sum(p == t for p, t in zip(y_pred, y_true))
total = len(y_true)
print(f"Accuracy: {100 * correct / total:.2f}%")


with torch.no_grad():
    for images, labels in mytest_loader:
        images, labels = images.to(device), labels.to(device)
        # print(images.shape)
        # print(labels)
        # with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print("Predicted:", predicted.cpu().numpy())
        print("True labels:", labels.cpu().numpy())
