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
# from facenet_pytorch import MTCNN
import sys

sys.path.append("C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo")


@st.cache
def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((112, 112)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    image = my_transforms(image).unsqueeze(0)
    return image


def load_lfw():
    """"""
    lfw_train = pd.read_csv(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\demo_train.csv')
    lfw_test = pd.read_csv(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\demo_test.csv')
    train_images = lfw_train['image1'].tolist() + lfw_train['image2'].tolist()
    test_images = lfw_test['image1'].tolist() + lfw_test['image2'].tolist()
    return train_images, test_images


def get_image_name(imge_list):
    image_names = []
    for path in imge_list:
        # Split the path on the backslash character (\)
        parts = path.split("\\")
        # Get the last part of the path (the file name)
        filename = parts[-1]
        image_names.append(filename)
        # Print the filename
    return image_names


if __name__ == '__main__':
    st.title('Face Recognition')
    st.write('This is a simple image classification web app to predict whether a face is real or fake.')
    logo = PIL.Image.open("C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\logo.png")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Add the image to the sidebar
    # Add the image to the bottom left corner of the app
    # file1 = st.file_uploader("Please upload the first image", type=["jpg", "png"])
    # file2 = st.file_uploader("Please upload the second image", type=["jpg", "png"])
    lfw_path = 'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\LFW_Demo'
    image_names = [image for image in os.listdir(lfw_path) if os.path.isfile(os.path.join(lfw_path, image))]
    train_images, test_images = load_lfw()
    train_image_names = get_image_name(train_images)
    test_image_names = get_image_name(test_images)
    people_train = [image[19:-9] for image in train_images]
    people_test = [image[19:-9] for image in test_images]
    people_train = list(set(people_train))
    people_test = list(set(people_test))

    if train_images:
        # select either train or test images
        image_type = st.radio('Select the image type', ('Train', 'Test'))
        if image_type == 'Train':
            image_names = train_image_names
            person_1 = st.sidebar.selectbox('Select the first person', people_train)
            person_2 = st.sidebar.selectbox('Select the second person', people_train)
        else:
            image_names = test_image_names
            person_1 = st.sidebar.selectbox('Select the first person', people_test)
            person_2 = st.sidebar.selectbox('Select the second person', people_test)


    image_1 = st.sidebar.selectbox('Select the first image', [image for image in image_names if person_1 in image])
    image_2 = st.sidebar.selectbox('Select the second image', [image for image in image_names if person_2 in image])
    image1 = PIL.Image.open(os.path.join(lfw_path, image_1))
    image2 = PIL.Image.open(os.path.join(lfw_path, image_2))
    st.image([image1, image2], caption=['Image 1', 'Image 2'], width=200)

    st.write("Classifying...")
    image1 = transform_image(image1)
    image2 = transform_image(image2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = load_predictor(
        "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\iresnet100_checkpoint.pth",
        device)
    embedder = Embedder(device=device, model_name='iresnet100', train=False)
    embedded_images = embedder(image1, image2)
    pred = predictor(embedded_images)[0]
    st.write("device: ", device)
    if pred == 1:
        st.markdown("The two images are of the **same** person ✅", )
        # st.write('The two images are of the same person.')
    else:
        st.markdown("The two images are **not** of the same person ❌")
        # st.write('The two images are of different people.')

    st.image(logo, width=100)
