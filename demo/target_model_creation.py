import torch
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
from Data.data_utils import convert_images_to_embedded_input, batch_test_prediction, evaluation, load_predictor
from FR_System.fr_system import FR_Api
from sklearn.model_selection import train_test_split
import sys
sys.path.append('C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\')

def load_LFW_pairs():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    lfw_train = LFWPairs(root='Data', transform=transform, download=False)
    return lfw_train


def target_model_creation(backbone_name, train_set, epoch_num, batch_size, device):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    # x_embedd_data = convert_images_to_embedded_input(train_loader, backbone_name, device)
    x_embedd_data = np.load(
        "C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\iresnet100_data_vectors.npy")
    x_train, x_test, y_train, y_test = train_test_split(x_embedd_data, train_set.targets, test_size=0.2, random_state=0)
    print(f'loading predictor model')
    predictor = load_predictor("C:\\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\Data\iresnet100_checkpoint.pth", device)

    # predictor = Predictor(predictor='NN', threshold=0.5, x_train=x_embedd_data, y_train=y_train, device=device,
    #                       batch_size=batch_size, epochs_num=epoch_num)
    embedder = Embedder(device=device, model_name=backbone_name, train=False)
    pred = predictor(x_test)
    print(evaluation(pred, y_test))


if __name__ == '__main__':
    # train_set = load_LFW_pairs()
    batch_size = 64
    epoch_num = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')
    # target_model_creation('iresnet100', train_set, epoch_num, batch_size, device)
