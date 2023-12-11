import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm


# ploting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# models 
import torchvision.models as models

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ------------------ READ ME ------------------ #

# save the model as "./best.ckpt" 

# ======= CHANGE FILE NAME HERE AND RUN ======= #
# MAKE SURE TO USE .jpg FILE
file_name = "./real1.jpg"
# ============================================= #

### model

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dimention [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128] 192
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64] 96

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32] 48

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16] 24

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8] 12
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4] 6 (img_size / (2 ** 5))
        )
        self.fc = nn.Sequential(
            nn.Linear(512*6*6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def show_image(image):
    plt.imshow(transforms.ToPILImage()(transforms.ToTensor()(image)), interpolation="bicubic")
    plt.show()

def extract_face(path):
    # face detector
    base_options = python.BaseOptions(model_asset_path='./detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    image = mp.Image.create_from_file(path)
    detection_result = detector.detect(image)

    image_copy = np.copy(image.numpy_view())
    rgb_annotated_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    faces = []
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
            
        face_region = rgb_annotated_image[y1:y2, x1:x2]
        faces.append(face_region) 

    size = (192 , 192 )  
    resized_faces = [cv2.resize(face, size) for face in faces]
    img = cv2.cvtColor(resized_faces[0], cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)
# transform for testing set
image_size = 192

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

train_tfm = transforms.Compose([
    transforms.Resize((image_size, image_size)),   
    transforms.RandomApply([   
        transforms.RandomApply([transforms.CenterCrop((160,160))], p=0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(10, expand=False, center=None, fill=256)], p=0.35),
        transforms.RandomApply([transforms.Pad(10, fill=256, padding_mode='constant')], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0)], p=1),
        transforms.RandomApply([transforms.ColorJitter(brightness=0, contrast=0.4, saturation=0, hue=0)], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(brightness=0, contrast=0, saturation=0.4, hue=0)], p=0.3),
        ],
    p=0.98),
    transforms.Resize((image_size, image_size)),  
    transforms.ToTensor(),
])

to_PIL = transforms.ToPILImage()


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

model = Classifier().to(device)
model.load_state_dict(torch.load(f"./best.ckpt", map_location=torch.device(device)))

image = extract_face(file_name)
show_image(image)

image = transform(image)
image = to_PIL(image)
image = transform(image)
image = image.unsqueeze(0)
pred = model(image.to(device))
pred = np.argmax(pred.cpu().data.numpy(), axis=1)
print(f"predicted : {pred[0]} | image is {'fake' if pred[0] == 0 else 'real'}")

