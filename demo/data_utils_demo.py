
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from torch import nn
from tqdm import tqdm
from embedder_demo import Embedder
from predictor_demo import Predictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def batch_test_prediction(fr, data_loder):
    pred = torch.tensor([], device=device)
    with torch.no_grad():
        for i, (image1, image2, label) in enumerate(tqdm(data_loder)):
            embedder = fr.embedder
            embedding = embedder(image1, image2)
            prediction = fr.predictor(embedding)
            if torch.is_tensor(prediction):
                pred = torch.cat((pred, prediction), 0)
            else:
                pred = torch.cat((pred, torch.tensor(prediction, device=device)), 0)
    pred = pred.detach().cpu().numpy().reshape((len(pred), 1))
    return pred


def evaluation(pred, labels):
    """
    The method evaluates results between predictions and labels.
    :param pred: Required. Type: ndarray. An array like object with the same dimensions as labels.
    :param labels: Required. Type: ndarray. An array like object with the same dimensions as pred.
    :return: dict. Evaluation results.
    """
    evaluation = {}
    labels = labels

    conf_mat = confusion_matrix(labels, pred)
    evaluation['tn'] = conf_mat[0][0]
    evaluation['fp'] = conf_mat[0][1]
    evaluation['fn'] = conf_mat[1][0]
    evaluation['tp'] = conf_mat[1][1]
    evaluation['acc'] = accuracy_score(labels, pred)
    evaluation['precision'] = precision_score(labels, pred)
    evaluation['recall'] = recall_score(labels, pred)
    evaluation['f1'] = f1_score(labels, pred)
    evaluation['auc'] = roc_auc_score(labels, pred)
    return evaluation


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
