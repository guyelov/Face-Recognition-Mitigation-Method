from PIL.Image import Image

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import FR_System.Embedder.embedder as embedder
import FR_System.Predictor.predictor as predictor
from torchvision import transforms
import Data.data_utils as data_utils

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((112, 112)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    image = my_transforms(image).unsqueeze(0)
    return image

if __name__ == '__main__':
    st.title('Face Recognition')
    st.write('This is a simple image classification web app to predict whether a face is real or fake.')
    file1 = st.file_uploader("Please upload the first image", type=["jpg", "png"])
    file2 = st.file_uploader("Please upload the second image", type=["jpg", "png"])
    if file1 is None or file2 is None:
        st.write('Please upload two images.')
    else:
        image1 = Image.open(file1)
        image2 = Image.open(file2)
        st.image(image1, caption='Uploaded Image.', use_column_width=True)
        st.image(image2, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor = load_predictor("C:\\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\Data\iresnet100_checkpoint.pth", device)
        embedder = Embedder(device=device, model_name='iresnet100', train=False)
        image1 = transform_image(image1)
        image2 = transform_image(image2)
        embedded_images = embedder(image1, image2)
        pred = predictor(embedded_images)
        if pred == 1:
            st.write('The two images are of the same person.')
        else:
            st.write('The two images are of different people.')