import sys

sys.path.append('C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\')
import torchvision.transforms as transforms
from torchvision.datasets import LFWPairs
from torch.utils.data import DataLoader, Dataset
from FR_System.Predictor.predictor import Predictor
from Data.data_utils import convert_images_to_embedded_input, evaluation, load_predictor, \
    batch_convert_data_to_net_input_from_multiple_backbones
from sklearn.model_selection import train_test_split

root = 'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\'


def load_LFW_pairs():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    lfw_train = LFWPairs(root=".", transform=transform, download=True)
    return lfw_train


def target_model_creation(backbone_name, train_set, epoch_num, batch_size, device):
    """
    create a predictor that trains on the embedding images from the embedder based on the embedder name
    :param backbone_name: Required. The name of the backbone to use.
    :param train_set: Required. The train set to train the predictor on.
    :param epoch_num: Required. The number of epochs to train the predictor on.
    :param batch_size: Required. The batch size to train the predictor on.
    :param device: Required. The device to train the predictor on can be 'cpu' or 'cuda'.
    :return: the accuracy of the predictor on the test set. and saves the predictor in the Data folder.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    x_embedd_data = convert_images_to_embedded_input(train_loader, backbone_name, device)
    # x_embedd_data = np.load("/content/drive/MyDrive/Data/iresnet100_data_vectors.npy") # CHANGE HERE
    x_train, x_test, y_train, y_test = train_test_split(x_embedd_data, train_set.targets, test_size=0.2, random_state=0)
    predictor = load_predictor("/content/drive/MyDrive/Data/iresnet100_checkpoint.pth", x_train=x_train,
                               y_train=y_train)  # CHANGE HERE
    predictor = Predictor(predictor='NN', threshold=0.5, x_train=x_train, y_train=y_train, device=device,
                          epochs_num=epoch_num, batch_size=batch_size,
                          nn_save_path="C:\\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data")
    pred = predictor(x_test)
    print(evaluation(pred, y_test))


def mitigation_target_model_creation(backbones_list, train_set, epoch_num, batch_size, device):
    """
    create a predictor that trains on the multiple backbones embedding images from the embedder based on the embedder list
    :param backbones_list: Required. The list of the backbones to use (for the mitigation method).
    :param train_set: Required. The train set to train the predictor on.
    :param epoch_num: Required. The number of epochs to train the predictor on.
    :param batch_size: Required. The batch size to train the predictor on.
    :param device: Required. The device to train the predictor on can be 'cpu' or 'cuda'.
    :return: the accuracy of the predictor on the test set. and saves the predictor in the Data folder.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    x_embedd_data = batch_convert_data_to_net_input_from_multiple_backbones(data_loder=train_loader,
                                                                            saving_path_and_name="C:\\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\mitigation_model",
                                                                            backbone_list=backbones_list)
    # x_embedd_data = np.load("/content/drive/MyDrive/Data/iresnet100_data_vectors.npy") # CHANGE HERE
    x_train, x_test, y_train, y_test = train_test_split(x_embedd_data, train_set.targets, test_size=0.2, random_state=0)
    # predictor = load_predictor("/content/drive/MyDrive/Data/iresnet100_checkpoint.pth",x_train=x_train,y_train=y_train) # CHANGE HERE
    predictor = Predictor(predictor='NN', threshold=0.5, x_train=x_train, y_train=y_train, device=device,
                          epochs_num=epoch_num, batch_size=batch_size,
                          nn_save_path="C:\\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data")
    pred = predictor(x_test)
    print(evaluation(pred, y_test))


if __name__ == '__main__':
    train_set = load_LFW_pairs()
