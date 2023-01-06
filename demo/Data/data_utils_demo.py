import os

import pandas as pd
import torch
from PIL.Image import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from FR_System.Embedder.embedder import Embedder
from FR_System.Predictor.predictor import Predictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = 'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\'


def transform_image(image):
    """
    This function transforms the image to the input of the backbone.
    :param image: Required. The image to transform.
    :return: The transformed image.
    """
    my_transforms = transforms.Compose([transforms.Resize((112, 112)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    image = my_transforms(image).unsqueeze(0)
    return image


def get_image_name(imge_list):
    """
    This function gets the image name from the image path.
    :param imge_list: Required. The image path list.
    :return: The image name list.
    """
    image_names = []
    for path in imge_list:
        # Split the path on the backslash character (\)
        parts = path.split("\\")
        # Get the last part of the path (the file name)
        filename = parts[-1]
        image_names.append(filename)
        # Print the filename
    return image_names


def convert_images_to_embedded_input(data_loader, backbone, device):
    """
    This function converts the data to the input of the backbone.
    :param data: Required. The data to convert.
    :param embedder_name: Required. The name of the backbone to use.
    :param saving_path_and_name: Required. The path and name to save the converted data.
    :return: The converted data.
    """
    data_vectors = []
    print(f'using the backbone: {backbone}')
    for i, (image1, image2, label) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            image1 = image1.to(device)
            image2 = image2.to(device)
            embedder = Embedder(device=device, model_name=backbone, train=False)
            embedding = embedder(image1, image2)
            data_vectors.append(embedding)
    data_vectors = np.vstack(data_vectors)
    np.save(
        f'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\{backbone}_data_vectors.npy',
        data_vectors)
    return data_vectors


def create_membership_pairs(train_set, attacker_set):
    """
    This function creates pairs of images from the same person.
    :param train_set: Required. The train set.
    :param train_set: Required. The attacker set.
    :param attacker_set: Required. The attacker set.
    :return:
    """
    names_train = []

    for index, row in train_set.iterrows():
        name1 = row['image1'].split('\\')[
            -1]  # split the file path by '' and then by '', and take all elements except the last (4-digit number)
        name2 = row['image2'].split('\\')[-1]  # same process for image2

        full_name1 = ''.join(name1)  # join first and last name with a space
        full_name2 = ''.join(name2)  # same for image2
        names_train.append(full_name1 + ' and ' + full_name2)  # combine the two names and append to the list
    names_attacker = []
    for index, row in attacker_set.iterrows():
        name1 = row['image1'].split('\\')[-1]
        name2 = row['image2'].split('\\')[-1]
        full_name1 = ''.join(name1)
        full_name2 = ''.join(name2)
        names_attacker.append(full_name1 + ' and ' + full_name2)
    return names_train, names_attacker




def load_predictor(traget_model_path, device='cpu'):
    """
    The function loads pre-trained predictor.
    :param traget_model_path: Required. str. The path stores weights of the target model.
    """
    n_in, n_out = 512, 1
    NN = nn.Sequential(
        nn.Linear(n_in, 512).to(device),
        nn.ReLU().to(device),
        nn.Linear(512, 256).to(device),
        nn.ReLU().to(device),
        nn.Linear(256, 128).to(device),
        nn.BatchNorm1d(128).to(device),
        nn.ReLU().to(device),
        nn.Linear(128, 64).to(device),
        nn.ReLU().to(device),
        nn.Linear(64, 32).to(device),
        nn.ReLU().to(device),
        nn.Linear(32, 16).to(device),
        nn.ReLU().to(device),
        nn.Linear(16, 8).to(device),
        nn.ReLU().to(device),
        nn.Linear(8, n_out).to(device),
        nn.Sigmoid().to(device))
    optimizer = torch.optim.Adam(NN.parameters(), lr=0.0001, weight_decay=0.0001)

    checkpoint = torch.load(traget_model_path, map_location=device)

    NN.load_state_dict(checkpoint['model_state_dict'])
    NN.eval()
    predictor = Predictor(predictor="NN", nn_instance=NN, threshold=0.5, device=device)
    return predictor


def demo_data_files():
    """
    This function creates the demo data files.
    :return: dict. The demo data files.
    """
    data_files = {}

    train_set = pd.read_csv(
        "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\target_model_demo_data_modified.csv")
    train_set = LFW_Demo(x_paris=train_set[["image1_modified", "image2_modified"]], y_paris=train_set["label"])
    attacker_set = pd.read_csv(
        "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\attack_demo_data.csv")
    test_set = LFW_Demo(x_paris=attacker_set[["image1", "image2"]], y_paris=attacker_set["label"])
    x_embedd_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\iresnet100_data_demo_vectors.npy")
    x_embedd_attacker_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\iresnet100_data_demo_attacker_vectors.npy")
    y_train = train_set.y
    y_test = test_set.y
    data_files['x_train'] = x_embedd_data
    data_files['y_train'] = y_train
    data_files['x_test'] = x_embedd_attacker_data
    data_files['y_test'] = y_test

    return data_files


def demo_mitigation_data_files():
    """
    This function creates the demo data files.
    :return: dict. The demo data files.
    """
    data_files = {}

    train_set = pd.read_csv(
        "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\target_model_demo_data_modified.csv")
    train_set = LFW_Demo(x_paris=train_set[["image1_modified", "image2_modified"]], y_paris=train_set["label"])
    attacker_set = pd.read_csv(
        "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\attack_demo_data.csv")
    test_set = LFW_Demo(x_paris=attacker_set[["image1", "image2"]], y_paris=attacker_set["label"])
    x_embedd_mitigation_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\mitigation_data_demo_vectors.npy")
    x_embedd_attacker_mitigation_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\mitigation_attacker_data_demo_vectors.npy")

    y_train = train_set.y
    y_test = test_set.y

    data_files['x_train'] = x_embedd_mitigation_data
    data_files['y_train'] = y_train
    data_files['x_test'] = x_embedd_attacker_mitigation_data
    data_files['y_test'] = y_test
    return data_files


class LFW_Demo(Dataset):
    """
    This class creates the dataset for the demo.
    """

    def __init__(self, x_paris, y_paris=None, transform=None):
        self.df = x_paris
        self.x_paris = x_paris
        if y_paris is not None:
            self.y = y_paris.values
        self.transform = transform

    def __getitem__(self, index):
        image_1 = self.df.iloc[index, 0]
        image_2 = self.df.iloc[index, 1]

        image_1 = Image.open(os.path.join(root, image_1))
        image_2 = Image.open(os.path.join(root, image_2))
        if self.transform is not None:
            image_1 = self.transform(image_1).to(device)
            image_2 = self.transform(image_2).to(device)
        if self.y is not None:
            return image_1, image_2, self.y[index]
        else:
            return image_1, image_2

    def __len__(self):
        return len(self.df)


def load_lfw(attack=False):
    """
    This function loads the LFW dataset.
    :param attack: bool. If True, the function loads the attack data.
    :return: dict. The LFW dataset.
    """
    """"""
    lfw_train = pd.read_csv(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\target_model_demo_data.csv')
    lfw_train['image1'] = lfw_train['image1'].apply(lambda x: x[19:])
    lfw_train['image2'] = lfw_train['image2'].apply(lambda x: x[19:])
    lfw_test = pd.read_csv(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\attack_demo_data.csv')
    train_images = lfw_train['image1'].tolist() + lfw_train['image2'].tolist()
    test_images = lfw_test['image1'].tolist() + lfw_test['image2'].tolist()
    if attack:
        lfw_attack = pd.read_csv(
            'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\attack_demo_data.csv')
        attack_images = lfw_attack['image1'].tolist() + lfw_attack['image2'].tolist()
        return train_images, attack_images
    return train_images, test_images


def names_process(names_train):
    """
    This function processes the names of the images.
    :param names_train: list. The names of the images.
    :return: list. The processed names of the images.
    """
    new_list = []
    for item in names_train:
        names = item.split(" and ")
        names = [name[:-4] for name in names]
        new_list.append(" and ".join(names))
    return new_list
