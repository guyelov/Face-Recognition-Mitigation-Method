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
    st.markdown("<div style='position: fixed; bottom: 0; left: 0;'>", unsafe_allow_html=True)
    st.image(logo, width=100)    file1 = st.file_uploader("Please upload the first image", type=["jpg", "png"])
    file2 = st.file_uploader("Please upload the second image", type=["jpg", "png"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image1 = PIL.Image.open(file1)
    image2 = PIL.Image.open(file2)


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
