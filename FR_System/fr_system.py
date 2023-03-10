from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import torch

from FR_System.Embedder.embedder import Embedder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
