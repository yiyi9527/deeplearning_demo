import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "../imgs/fashion02.png"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                            torchvision.transforms.Grayscale(num_output_channels=1),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

# 搭建神经网络
class AlexNet_Model(nn.Module):
    def __init__(self):
        super(AlexNet_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Flatten(),
            nn.Linear(6400,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

model = torch.load("alexnet_10.pth",map_location=torch.device("cpu"))
print(model)

image = torch.reshape(image,[1,1,224,224])
model.eval()
with torch.no_grad():
    output = model(image)

print(output)
print(output.argmax(1))