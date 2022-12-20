from tqdm import tqdm

from FR_System.Embedder.embedder import Embedder, process_image
# from FR_System.Embedder.embedder_tensor import Embedder, process_image
from FR_System.Predictor.predictor_tensor import Predictor
# from FR_System.Predictor.predictor import Predictor
from Defenses.Backbone_Generate import predict_multiple_backbones, embedder_generator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_prediction(x_test, fr, multiple_backbones=False, seed=None):
    """
    Get the FR system for the test set given.
    :param x_test: Required. Type: Dataframe. Dataframe of pairs path.
    :param fr: Required. Type: FR_System. The used face recognition system.
    :return: ndarray of the predictions.
    """
    pred = []
    for i, row in x_test.iterrows():
        path1 = row["path1"]
        path2 = row["path2"]
        np_image1 = process_image(path1)
        np_image2 = process_image(path2)
        if multiple_backbones:
            prediction = predict_multiple_backbones(np_image1, np_image2, fr, i, seed=seed)
        else:
            prediction = fr.predict(np_image1, np_image2)
        if torch.is_tensor(prediction):
            pred.append(prediction.detach().numpy()[0])
        else:
            pred.append(prediction)
    pred = np.asarray(pred).reshape((len(pred), 1))
    return pred


def batch_test_prediction(fr, data_loder, multiple_backbones=False, seed=None, backbone_list=None):
    """
    This method is used to test the FR system on a batch of data.
    :param fr: Required. Type: FR_System. The used face recognition system.
    :param data_loder: Required. Type: DataLoader. The data loader for the test set.
    :param multiple_backbones: Optional. Type: bool. If True, the FR system will use multiple backbones.
    :param seed: Optional. Type: int. The seed for the random number generator.
    :return: ndarray of the predictions.
    """
    # pred = []
    if multiple_backbones:
        backbones_list = embedder_generator(seed, backbone_list)
    pred = torch.tensor([], device=device)
    with torch.no_grad():
        for i, (image1, image2, label) in enumerate(tqdm(data_loder)):
            if multiple_backbones:
                backbone_index = (i + len(backbones_list)) % len(backbones_list)
                embedder = Embedder(device=device, model_name=backbones_list[backbone_index], train=False)
                embedding = embedder(image1, image2)
                prediction = fr.predictor(embedding)
            else:
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
    labels = labels.astype(int)
    pred = pred.astype(int)

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


class FR_Api(torch.nn.Module):
    """
    The face recognition API class.
    """

    def __init__(self, embedder, predictor):
        """
        Constructor.
        :param embedder: Required. Type: Embeder object.
        :param predictor: Required. Type: Predictor object.
        """
        super(FR_Api, self).__init__()
        # assert isinstance(embedder, Embedder)
        # assert isinstance(predictor, Predictor)
        self.embedder = embedder
        self.predictor = predictor


    def forward(self, image1, image2):
        """
        The method predicts whether the two images are of the same person.
        :param image1: Required. Type: ndarray / torch.tensor. An array like object with 4 dimensions:
                        (batch size, channels, image hight, image width).
                        For example: (24, 3, 112, 112).
        :param image2: Required. Type: ndarray / torch.tensor. An array like object with 4 dimensions:
                        (batch size, channels, image hight, image width).
                        For example: (24, 3, 112, 112).
        :return: the probability of them to be the same person.
        """
        embedding = self.embedder(image1, image2)
        prediction = self.predictor(embedding)
        return prediction
    # def forward(self, x):
    #     ''' Forward pass of the model. '''
    #     prediction_array = []
    #     for i in x:
    #         image_pair = i
    #         image_1, image_2 = image_pair[0], image_pair[1]
    #
    #         prob = True
    #
    #         emb_pred = self.embedder.embedder(image_1)
    #         emb_pred2 = self.embedder.embedder(image_2)
    #         res_of_sub = torch.subtract(emb_pred, emb_pred2)
    #
    #         y = self.predictor.nn(res_of_sub)
    #         y_comp = 1.0 - y
    #         y_two_class = torch.cat((y_comp, y), -1)
    #         prediction_array.append(y_two_class)
    #     #####
    #     # y_comp= 1.0-y
    #     # y_two_class = torch.cat((y_comp, y), -1)
    #     #####
    #     # return y_two_class
    #
    #     return torch.cat(prediction_array)
    #     # return torch.FloatTensor(prediction_array)
