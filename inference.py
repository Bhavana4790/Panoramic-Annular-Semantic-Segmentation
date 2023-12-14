import numpy as np
import torch
import os
import importlib
import types
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize #Scale,
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet_pspnet import Net
from transform import Relabel, ToLabel, Colorize

import visdom

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024*1),Image.BILINEAR),
    ToTensor(),
])

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
      own_state = model.state_dict()
      
      for a in own_state.keys():
          print(a)
      for a in state_dict.keys():
          print(a)
      print('-----------')
      
      for name, param in state_dict.items():
          if name not in own_state:
                continue
          own_state[name].copy_(param)
      
      return model

def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    input_tensor = input_transform_cityscapes(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor


def Segmentation(image):
    # Preprocess the input image
    input_tensor = preprocess_image(image)
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    label = output[0].cpu().max(0)[1].data.byte()
    label_color = Colorize()(label.unsqueeze(0))
    label_save = ToPILImage()(label_color)
    # label_save.save("./results/result1.png")
    label_np = np.array(label_save)
    return label_np

# image_path = r'C:\Users\Admin\Downloads\Panomatic_segmentation\0.jpg'
# Segmentation(image_path)
NUM_CHANNELS = 3
NUM_CLASSES = 28

model_path = 'erfpspnet.pth'
model = Net(NUM_CLASSES)
model = torch.nn.DataParallel(model)
model = load_my_state_dict(model, torch.load(model_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()
