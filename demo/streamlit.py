import os

import PIL.Image
import streamlit as st
import pandas as pd
import numpy as np
import torch
from embedder_demo import Embedder
from predictor_demo import Predictor
from torchvision import transforms
from data_utils_demo import load_predictor
from facenet_pytorch import MTCNN


def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((112, 112)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    image = my_transforms(image).unsqueeze(0)
    return image


if __name__ == '__main__':
    st.title('Face Recognition')
    st.write('This is a simple image classification web app to predict whether a face is real or fake.')
    logo = PIL.Image.open("demo/logo.png")

    # Add the image to the sidebar
    # Add the image to the bottom left corner of the app
    file1 = st.file_uploader("Please upload the first image", type=["jpg", "png"])
    file2 = st.file_uploader("Please upload the second image", type=["jpg", "png"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lfw_path = "demo/Data/lfw-py/lfw_funneled"
    if lfw_path:
        # List the images in the LFW folder
        image_files = [f for f in os.listdir(lfw_path) if f.endswith(".jpg") or f.endswith(".png")]
        file1_select = st.selectbox("Select the first image from the LFW folder:", image_files)
        file2_select = st.selectbox("Select the second image from the LFW folder:", image_files)
        # Load the selected images from the LFW folder
        image1 = PIL.Image.open(os.path.join(lfw_path, file1_select))
        image2 = PIL.Image.open(os.path.join(lfw_path, file2_select))
    if file1 is None and file2 is None and not lfw_path:
        st.write("Please upload two face images")
    else:
        if file2 is None and not lfw_path:
            st.write("Please upload the second image")
        else:
            # image1 = PIL.Image.open(file1)
            # image2 = PIL.Image.open(file2)
            #

            st.image(image1, caption='Uploaded Image.', use_column_width=True)
            st.image(image2, caption='Uploaded Image.', use_column_width=True)

            st.write("Classifying...")
            image1 = transform_image(image1)
            image2 = transform_image(image2)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            predictor = load_predictor("demo/iresnet100_checkpoint.pth", device)
            embedder = Embedder(device=device, model_name='iresnet100', train=False)
            embedded_images = embedder(image1, image2)
            pred = predictor(embedded_images)[0]
            st.write(predictor(embedded_images, return_proba=True))
            if pred == 1:
                st.write('The two images are of the same person.')
            else:
                st.write('The two images are of different people.')
    st.image(logo, width=100)
