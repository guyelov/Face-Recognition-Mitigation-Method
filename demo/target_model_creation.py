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
from Data.data_utils import convert_images_to_embedded_input
from FR_System.fr_system import FR_Api


def load_LFW_pairs():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    lfw_train = LFWPairs(root='Data', split='train', transform=transform, download=True)
    lfw_test = LFWPairs(root='Data', split='test', transform=transform, download=True)
    return lfw_train, lfw_test


def target_model_creation(backbone_name, train_set, test_set, batch_size, device):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    x_embedd_data = convert_images_to_embedded_input(train_loader, backbone_name, device)
    predictor = Predictor(predictor='NN', threshold=0.5, x_train=x_embedd_data, y_train=train_set.targets, device=device)
    embedder = Embedder(device=device, model_name=backbone_name, train=False)
    fr = FR_Api(embedder, predictor)


if __name__ == '__main__':
    train_set, test_set = load_LFW_pairs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_model_creation('iresnet100', train_set, test_set, 64, device)
