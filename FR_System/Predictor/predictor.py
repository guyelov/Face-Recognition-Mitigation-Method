import os
import numpy as np
import torch
from torch import nn
import gc
from pathlib import Path


class Predictor(nn.Module):
    """
    The predictor class the predictor is used to predict whether two images are of the same person.
    it can be used with different predictors, such as cosine similarity, euclidean distance, or a neural network.
    """

    def __init__(self, predictor=None, x_train=None, y_train=None,
                 nn_save_path="", nn_instance=None, threshold=0.5, embeder=None, device="cpu", n_in=512,
                 pretrained_predicor=None,
                 batch_size=64, epochs_num=30):
        """
        The constructor of the predictor class.
        :param predictor: Optional. Type: str. The type of predictor to use.
                          Options: "cosine similarity", "Euclidean distance", "Chi-Square distance",
                          "Bhattacharyya distance", "NN".
        :param x_train: Optional. Type: ndarray. The training data for the NN. Must be provided in case
                        predictor="NN" and nn_instance=None.
        :param y_train: Optional. Type: ndarray. The training labels for the NN. Must be provided in case
                        predictor="NN" and nn_instance=None.
        :param nn_save_path: Optional. Type: str. The saving path of the trained NN.
        :param nn_instance:  Optional. Type: torch model. Instance of NN to use as the predictor.
        :param threshold: Optional. Type: float between [0,1]. The threshold for the predictions.
                          If None, the probability will be returned.
        """
        super(Predictor, self).__init__()

        self.predictor = predictor
        self.device = device
        self.embedder = embeder
        self.nn_instance = None
        if pretrained_predicor:
            self.pretrained_predicor = pretrained_predicor

        if predictor == "NN":
            if nn_instance is None:
                assert (x_train is not None)
                assert (y_train is not None)
                self.nn = self.train_NN(x_train, y_train, saving_path=nn_save_path, embeder=self.embedder,
                                        n_in=n_in, epoch_num=epochs_num, batch_size=batch_size)
            else:
                self.nn_instance = nn_instance
                self.nn = nn_instance
        self.threshold = threshold

    def train_NN(self, x_train, y_train, lossf=torch.nn.BCEWithLogitsLoss(), batch_size=128, epoch_num=40,
                 lr=0.0001, saving_path="", embeder=None, n_in=None):
        """
        Train an NN to use as a predictor.
        :param x_train: Required. Type: ndarray/torch tensor. The training data for the NN.
        :param y_train: Required. Type: ndarray/torch tensor. The training labels for the NN.
        :param lossf: Optional. The loss function instance to use. If not given, it is equal to cross entropy loss.
        :param batch_size: Optional. The batch size to use during training. If not given, it is equal to 64.
        :param epoch_num: Optional. The number of epoch to do during training. If not given, it is equal to 10.
        :param lr: Optional. Type: float. The learning rate used durring training. If not given, it is equal to 0.0001.
        :param saving_path: Optional. Type: str. The location to save the checkpoints and complete net.
        :return: nn.Sequensial a trained NN.
        """
        assert x_train.shape[0] == len(y_train)

        torch.manual_seed(0)
        np.random.seed(0)
        if n_in is None:
            n_in = x_train.shape[1]
        n_out = 1
        if self.nn_instance:
            model = self.nn_instance
        else:
            model = nn.Sequential(
                nn.Linear(n_in, 512).to(self.device),
                nn.ReLU().to(self.device),
                nn.Linear(512, 256).to(self.device),
                nn.ReLU().to(self.device),
                nn.Linear(256, 128).to(self.device),
                nn.BatchNorm1d(128).to(self.device),
                nn.ReLU().to(self.device),
                nn.Linear(128, 64).to(self.device),
                nn.ReLU().to(self.device),
                nn.Linear(64, 32).to(self.device),
                nn.ReLU().to(self.device),
                nn.Linear(32, 16).to(self.device),
                nn.ReLU().to(self.device),
                nn.Linear(16, 8).to(self.device),
                nn.ReLU().to(self.device),
                nn.Linear(8, n_out).to(self.device),
                nn.Sigmoid().to(self.device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        epoch = 0

        model.train()
        if epoch == epoch_num - 1:
            model.eval()
            return model
        for epoch in range(epoch, epoch_num):
            for i in range(0, x_train.shape[0], batch_size):
                optimizer.zero_grad()
                batch_x = x_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                input_x = batch_x
                y_pred = model(torch.tensor(input_x).float().to(self.device))
                loss = lossf(y_pred.float().flatten(),
                             torch.tensor(batch_y, device=self.device, dtype=torch.float))
                loss.backward()
                optimizer.step()
                self.embedder = embeder
                # print('epoch: ', epoch, ' part: ', i, ' loss: ', loss.item())

            gc.collect()
            torch.cuda.empty_cache()

            print('epoch: ', epoch, ' loss: ', loss.item())
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'loss': loss},
                   saving_path + "checkpoint.pth")
        model.eval()
        return model

    def net(self, vector, return_proba, art):
        """
        The method returns the probability for class 1 according to the trained NN.
        :param vector1: Required. Type: ndarray/torch tensor. Image vector 1
        :param vector2: Required. Type: ndarray/torch tensor. Image vector 2
        :param return_proba: Optional. Type: boolean. Whether to return the probability. Default is False.
        :return:
        """
        if art:
            proba = self.nn(torch.tensor(vector).float().to(self.device))
            return proba
        if torch.is_tensor(vector):
            diff = vector
            proba = self.nn(torch.tensor(diff, device=self.device).float())
        else:
            diff = vector
            proba = self.nn(torch.tensor(diff, device=self.device).float())
        if return_proba:
            # return proba.flatten().tolist()[0]
            return proba.flatten().tolist()

        else:
            lst = list(map(int, (proba >= self.threshold).reshape(-1)))
            return list(map(int, (proba >= self.threshold).reshape(-1)))

    def forward(self, vector, return_proba=False, art=True):
        return self.net(vector, return_proba, art)

    def load_checkpoint(self, path, model, optimizer):
        """
        The method loads the last checkpoint from the given path.
        :param path: Required. Type: str. The path to the checkpoint.
        :param model: Required. Type: nn.Sequential. The model to load.
        :param optimizer: Required. Type: torch.optim. The optimizer to load.
        :param embedder: Optional. Type: nn.Sequential. The embedder to load.
        :return: The model, optimizer and the epoch number.
        """

        preditor_path = "{}checkpoints".format(path)
        last_checkpoint = sorted(Path(preditor_path).iterdir(), key=os.path.getmtime, reverse=True)[0]
        checkpoint = torch.load(last_checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        model.train()
        return model, optimizer, epoch
