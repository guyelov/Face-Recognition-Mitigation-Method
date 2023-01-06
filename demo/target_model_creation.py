import os
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
import pandas as pd
import torch
import sys

from sklearn.metrics import accuracy_score

sys.path.append('C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import LFWPairs
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from FR_System.Embedder.embedder import Embedder
from FR_System.Predictor.predictor import Predictor
from Data.data_utils import convert_images_to_embedded_input, batch_test_prediction, evaluation, load_predictor, \
    batch_convert_data_to_net_input_from_multiple_backbones
from FR_System.fr_system import FR_Api
from sklearn.model_selection import train_test_split

root = 'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\'


def load_LFW_pairs():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    lfw_train = LFWPairs(root='Data', transform=transform, download=False)
    return lfw_train


train_set = pd.read_csv(
    "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\target_model_demo_data.csv")
attacker_set = pd.read_csv(
    "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\attack_demo_data.csv")

# Load the dataframe
df = train_set

# Create a new column for the modified paths
df["image1_modified"] = df["image1"].str.replace(r"demo\\Data\\LFW_Demo\\demo\\Data\\LFW_Demo\\",
                                                 r"demo\\Data\\LFW_Demo\\")
df["image2_modified"] = df["image2"].str.replace(r"demo\\Data\\LFW_Demo\\demo\\Data\\LFW_Demo\\",
                                                 r"demo\\Data\\LFW_Demo\\")

# Save the modified dataframe to a new file
train_set = df[["image1_modified", "image2_modified", "label"]]
train_set.to_csv(
    "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\target_model_demo_data_modified.csv",
    index=False)


class LFW_Demo(Dataset):
    """Custom Dataset for loading LFW Demo face images"""

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


def target_model_creation(backbone_name, train_set, epoch_num, batch_size, device, test_set=None,weight_decay=0.0005,lr=0.001):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    backbones_list = ['ResNet152-irse', 'ResNet50-ir', 'iresnet100']
    y_train = train_set.y
    y_test = test_set.y

    # x_embedd_mitigation_data = batch_convert_data_to_net_input_from_multiple_backbones(data_loder=train_loader,
    #                                                                                    saving_path_and_name=r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\iresnet100_data_demo_vectors.npy"
    #                                                                                    , seed=0,
    #                                                                                    backbone_list=backbones_list)
    # x_embedd_mitigation_attacker_data = batch_convert_data_to_net_input_from_multiple_backbones(data_loder=test_loader,
    #                                                                                             saving_path_and_name=r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\mitigation_attacker_data_demo_vectors.npy",
    #                                                                                             seed=0,
    #                                                                                             backbone_list=backbones_list)
    x_embedd_mitigation_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\iresnet100_data_demo_vectors.npy")
    y_train = train_set.y
    # predictor_mitigation = Predictor(x_train=x_embedd_mitigation_data, y_train=y_train, device=device,
    #                                  batch_size=batch_size,
    #                                  nn_save_path=r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\Data\iresnet100_mitigation_demo_checkpoint.pthcheckpoint.pth",
    #                                  predictor="NN", epochs_num=epoch_num)

    x_embedd_attacker_mitigation_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\mitigation_attacker_data_demo_vectors.npy")
    predictor_mitigation = load_predictor(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\Data\iresnet100_mitigation_demo_checkpoint.pthcheckpoint.pthcheckpoint.pth",
        device)

    pred_mitigation = predictor_mitigation(x_embedd_mitigation_data).detach().cpu().numpy()
    pred_mitigation_attacker = predictor_mitigation(x_embedd_attacker_mitigation_data).detach().cpu().numpy()
    # print("Mitigation model accuracy on train set: ", accuracy_score(y_train, pred_mitigation))
    # print("Mitigation model accuracy on test set: ", accuracy_score(y_test, pred_mitigation_attacker))

    x_embedd_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\iresnet100_data_demo_vectors.npy")
    x_embedd_attacker_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\iresnet100_data_demo_attacker_vectors.npy")
    x_embedd_mitigation_data = np.load(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\mitigation_data_demo_vectors.npy")
    # predictor = Predictor(x_train=x_embedd_data, y_train=y_train, device=device,
    #                       batch_size=256,
    #                       nn_save_path=r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\Data\iresnet100_demo_checkpoint",
    #                       predictor="NN")
    predictor = load_predictor(
        r"C:\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\Data\iresnet100_demo_checkpointcheckpoint.pth",
        device)

    y_train = train_set.y
    pred_mitigation = predictor_mitigation(x_embedd_mitigation_data).detach().cpu().numpy()
    pred_attacker_mitigation = predictor_mitigation(x_embedd_attacker_mitigation_data).detach().cpu().numpy()
    embedder = Embedder(model_name=backbone_name, device=device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(predictor.nn.parameters()) + list(embedder.parameters()), lr=0.0001,
                                 weight_decay=0.0001)
    y_test = test_set.y
    target_model = PyTorchClassifier(model=predictor, loss=criterion, optimizer=optimizer, input_shape=(512,),
                                     nb_classes=2)
    train_pred = np.array([np.round(arr) for arr in target_model.predict(x_embedd_data.astype(np.float32))])
    test_pred = np.array([np.round(arr) for arr in target_model.predict(x_embedd_attacker_data.astype(np.float32))])

    print('Base model Train accuracy: ', accuracy_score(y_train, train_pred))
    print('Base model Train accuracy: ', accuracy_score(y_test, test_pred))

    predictions = predictor(x_embedd_data).detach().cpu().numpy()
    test_predictions = predictor(x_embedd_attacker_data).detach().cpu().numpy()
    mlp_attack_bb = MembershipInferenceBlackBox(target_model, attack_model_type='rf')
    # mlp_attack_bb.fit(x_train, y_train, x_test, y_test)

    mlp_attack_bb.fit(pred=predictions, y=y_train, test_pred=test_predictions, test_y=y_test, x=None, test_x=None)
    mlp_inferred_train_bb = mlp_attack_bb.infer(x_embedd_data, y_train)
    mlp_inferred_test_bb = mlp_attack_bb.infer(x_embedd_attacker_data, y_test)
    mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
    mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
    mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (
            len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))
    print(f"Members Accuracy: {mlp_train_acc_bb:.4f}")
    print(f"Non Members Accuracy {mlp_test_acc_bb:.4f}")
    print(f"Attack Accuracy {mlp_acc_bb:.4f}")
    print(
        f'___________________________________Testing Mitigation Model______________________________________________________')
    backbones_list = ['ResNet152-irse', 'ResNet50-ir', 'iresnet100']

    embedder_1 = Embedder(model_name=backbones_list[0], device=device)
    embedder_2 = Embedder(model_name=backbones_list[1], device=device)
    embedder_3 = Embedder(model_name=backbones_list[2], device=device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(predictor.nn.parameters()) , lr=lr,
                                 weight_decay=weight_decay)

    target_model_mitigation = PyTorchClassifier(model=predictor_mitigation, loss=criterion, optimizer=optimizer,
                                                input_shape=(512,),
                                                nb_classes=2)
    train_pred = np.array(
        [np.round(arr) for arr in target_model_mitigation.predict(x_embedd_mitigation_data.astype(np.float32))])
    test_pred = np.array([np.round(arr) for arr in
                          target_model_mitigation.predict(x_embedd_attacker_mitigation_data.astype(np.float32))])

    print('Base model Train accuracy: ', accuracy_score(y_train, train_pred))
    print('Base model Train accuracy: ', accuracy_score(y_test, test_pred))

    mlp_attack_bb_mitiagtion = MembershipInferenceBlackBox(target_model_mitigation, attack_model_type='rf')
    # mlp_attack_bb.fit(x_train, y_train, x_test, y_test)
    mlp_attack_bb_mitiagtion.fit(pred=pred_mitigation, y=y_train, test_pred=pred_attacker_mitigation, test_y=y_test,
                                 x=None, test_x=None)
    mlp_inferred_train_bb = mlp_attack_bb_mitiagtion.infer(x_embedd_mitigation_data, y_train)
    mlp_inferred_test_bb = mlp_attack_bb_mitiagtion.infer(x_embedd_attacker_mitigation_data, y_test)
    mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
    mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
    mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (
            len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))
    print(f"Members Accuracy: {mlp_train_acc_bb:.4f}")
    print(f"Non Members Accuracy {mlp_test_acc_bb:.4f}")
    print(f"Attack Accuracy {mlp_acc_bb:.4f}")


if __name__ == '__main__':
    pass
    # train_set = load_LFW_pairs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')
    epoch_num = 90
    transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    train_set = LFW_Demo(x_paris=train_set[["image1_modified", "image2_modified"]], y_paris=train_set["label"],
                         transform=transforms)
    test_set = LFW_Demo(x_paris=attacker_set[["image1", "image2"]], y_paris=attacker_set["label"], transform=transforms)
    batch_list = [70,80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    epoch_list = [40,50,60,70,80,90,100]
    target_model_creation('iresnet100', train_set, 90, 80, device, test_set=test_set,
                          weight_decay=0.0001, lr=0.0001)


