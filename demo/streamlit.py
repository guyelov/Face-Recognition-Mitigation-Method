import PIL.Image
import streamlit as st
import pandas as pd
import numpy as np
import torch
from embedder_demo import Embedder
from predictor_demo import Predictor

from torchvision import transforms
from data_utils_demo import load_predictor
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
        image1 = PIL.Image.open(file1)
        image2 = PIL.Image.open(file1)
        st.image(image1, caption='Uploaded Image.', use_column_width=True)
        st.image(image2, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor = load_predictor("demo/iresnet100_checkpoint.pth", device)
        embedder = Embedder(device=device, model_name='iresnet100', train=False)
        image1 = transform_image(image1)
        image2 = transform_image(image2)
        embedded_images = embedder(image1, image2)
        st.write(embedded_images)
        pred = predictor(embedded_images)[0]
        st.write(pred[0])
        if pred == 1:
            st.write('The two images are of the same person.')
        else:
            st.write('The two images are of different people.')