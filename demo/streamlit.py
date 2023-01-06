import os
import PIL.Image
import streamlit as st
import sys

sys.path.append("C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method")
from Attacks.Membership_Inference.MembershipAttack import black_box_membership_attack
from demo.Data.data_utils_demo import load_predictor, get_image_name, transform_image, demo_data_files, load_lfw, \
    demo_mitigation_data_files, names_process
from FR_System.Embedder.embedder import Embedder
import torch
import numpy as np


@st.cache
def predict(image_1, image_2, art=True):
    """
    Given two images, this function returns a prediction on whether the two images are of the same person or not.
    :param image_1: PIL image
    :param image_2: PIL image
    :param art: (bool, optional): Whether to return the prediction in ART format (True) or in PyTorch format (False).
    :return: prediction float: A prediction value between 0 and 1, where 0 indicates that the images are not similar and 1 indicates that the images are similar.

    """
    image1 = transform_image(image_1)
    image2 = transform_image(image_2)
    embedded_images = embedder(image1, image2)
    pred = predictor(embedded_images, art=art)[0]
    return pred


def face_verification_demo():
    """
    This function demonstrates a simple image classification task to predict whether two images are of the same person or not.
    The user has the option to either upload their own images or choose from the LFW (Labeled Faces in the Wild) dataset
    If the user chooses to upload their own images, they can select two images and the model will classify whether they are of the same person or not.
    If the user chooses to select images from the LFW dataset, they can choose between train and test images and select two images of different people.
    The function then displays the selected images and classifies whether they are of the same person or not.
    :return: None
    """
    st.write(
        'This is a simple image classification demo app to predict whether two images are of the same person or not.')
    st.write('You can upload two images and the model will predict whether they are of the same person or not.')
    st.write('Or you can choose two images from the LFW dataset.')

    option = st.selectbox('Choose your option', ('Upload your own images', 'Choose from LFW dataset'))
    if option == 'Upload your own images':
        file1 = st.file_uploader("Please upload the first image", type=["jpg", "png"])
        file2 = st.file_uploader("Please upload the second image", type=["jpg", "png"])
        if file1 is not None and file2 is not None:
            image1 = PIL.Image.open(file1)
            image2 = PIL.Image.open(file2)
            st.image([image1, image2], caption=['Image 1', 'Image 2'], width=200)
            pred = predict(image1, image2, art=False)
            st.write("Classifying...")
            if pred == 1:
                st.markdown("The two images are of the **same** person ✅", )
            else:
                st.markdown("The two images are **not** of the same person ❌")


    else:

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

        st.markdown("Classifying...:hourglass_flowing_sand:")

        pred = predict(image1, image2, art=False)

        if pred == 1:
            st.markdown("The two images are of the **same** person ✅", )
            # st.write('The two images are of the same person.')
        else:
            st.markdown("The two images are **not** of the same person ❌")
            # st.write('The two images are of different people.')


def intro():
    st.markdown('Welcome to the Face Recognition Mitigation Method Hackathon Demo!')
    st.markdown('In this demo we will represent our face verification system and Membership Inference Attack demo.')
    st.markdown('Choose one of the options from the sidebar to get started.')
    st.markdown(
        'For more information about the project, please visit our [GitHub repository](https://github.com/guyelov/Face-Recognition-Mitigation-Methods).')


def membership_inference_attack(names_train, names_attacker, lfw_path):
    """
    This function demonstrates a membership inference attack on the face verification system with mitigation method and without mitigation method.
    :param names_train: (list): A list of names of the people in the training set.
    :param names_attacker: (list): A list of names of the people in the attacker set.
    :param lfw_path: (str): The path to the LFW dataset.
    :return: None
    """
    image_type = st.radio('Select the image type', ('Train', 'Test'))
    names_processed = names_process(names_train)

    if image_type == 'Train':
        pair = st.sidebar.selectbox('Select the pair', names_processed)
        pair = names_train[names_processed.index(pair)]
        image1 = pair.split(' and ')[0]
        image2 = pair.split(' and ')[1]
        if image1[:-9] == image2[:-9]:
            label = 1
        else:
            label = 0
        image1 = PIL.Image.open(os.path.join(lfw_path, image1))
        image2 = PIL.Image.open(os.path.join(lfw_path, image2))

    else:
        pair = st.sidebar.selectbox('Select the pair', names_attacker)
        image1 = pair.split(' and ')[0]
        image2 = pair.split(' and ')[1]
        if image1[:-9] == image2[:-9]:
            label = 1
        else:
            label = 0
        image1 = PIL.Image.open(os.path.join(lfw_path, image1))
        image2 = PIL.Image.open(os.path.join(lfw_path, image2))

    st.image([image1, image2], caption=['Image 1', 'Image 2'], width=200)
    membership_model = attack_no_mitigation
    mitigation_model = attack_mitigation
    image1 = transform_image(image1)
    image2 = transform_image(image2)
    embedded_images = embedder(image1, image2)

    inference = membership_model.infer(embedded_images, np.array([label]))
    mitigation_inference = mitigation_model.infer(embedded_images, np.array([label]))
    st.write("Classifying...")
    if inference == 1:
        st.markdown("These images were **used** to train the model ✅", )
    else:
        st.markdown("These images were **not** used to train the model ❌")
    st.markdown("Testing the attack when using the mitigation method:")

    if mitigation_inference == 1:
        st.markdown("These images were **used** to train the model ✅", )
    else:
        st.markdown("These images were **not** used to train the model ❌")


def membership_demo():
    st.markdown('Welcome to the Membership Inference Attack Demo!')
    st.markdown('In this demo we will represent an existing Membership Inference Attack')
    st.markdown("We will perform the attack on our FR model with the mitigation method and without it")

    names_train = np.load(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\names_train.npy')
    names_attacker = np.load(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\names_attacker.npy')

    membership_inference_attack(names_train, names_attacker, lfw_path)


if __name__ == '__main__':
    st.set_page_config(page_title='Face Recognition Mitigation Demo', page_icon=':camera:',
                       layout='centered', initial_sidebar_state='auto')

    lfw_path = 'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\LFW_Demo'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictor = load_predictor(
        "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\iresnet100_checkpoint.pth",
        device)
    embedder = Embedder(device=device, model_name='iresnet100', train=False)
    data_dict = demo_data_files()
    x_train, x_test, y_train, y_test = data_dict['x_train'], data_dict['x_test'], data_dict['y_train'], data_dict[
        'y_test']
    predictor_attack = load_predictor(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\Data\iresnet100_demo_checkpointcheckpoint.pth",
        device)
    predictor_mitigation = load_predictor(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\Data\iresnet100_mitigation_demo_checkpoint.pthcheckpoint.pthcheckpoint.pth",
        device)
    data_mitigation_dict = demo_mitigation_data_files()
    x_train_mitigation, x_test_mitigation, y_train_mitigation, y_test_mitigation = data_mitigation_dict['x_train'], \
                                                                                   data_mitigation_dict['x_test'], \
                                                                                   data_mitigation_dict['y_train'], \
                                                                                   data_mitigation_dict[
                                                                                       'y_test']
    attack_no_mitigation = black_box_membership_attack(predictor_attack, embedder, x_train, y_train, x_test, y_test)
    attack_mitigation = black_box_membership_attack(predictor_mitigation, embedder, x_train_mitigation,
                                                    y_train_mitigation,
                                                    x_test_mitigation, y_test_mitigation)

    st.title('Face Recognition Mitigation Method Hackathon Demo')
    st.sidebar.title('Face Recognition')
    st.sidebar.markdown('Choose one of the options from the sidebar to get started.')
    page_names_to_functions = {
        'About': intro,
        'Face Verification': face_verification_demo,
        'Membership Inference Attack': membership_demo
    }
    page_name = st.sidebar.radio('Navigation', list(page_names_to_functions.keys()))
    page_names_to_functions[page_name]()
