import torch
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier


def black_box_membership_attack(predictor, embedder, x_train, y_train, x_test, y_test):
    """
    This function creates a black box membership attack.
    :param predictor: The target model.
    :param embedder: The embedder model.
    :param x_train: The training data.
    :param y_train: The training labels.
    :param x_test: The testing data.
    :param y_test: The testing labels.
    :return: The black box membership attack model.
    """
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(list(predictor.nn.parameters()) + list(embedder.parameters()), lr=0.0001,
                                 weight_decay=0.0001)

    target_model = PyTorchClassifier(model=predictor, loss=criterion, optimizer=optimizer, input_shape=(512,),
                                     nb_classes=2)
    predictions = predictor(x_train).detach().numpy()
    test_predictions = predictor(x_test).detach().numpy()
    mlp_attack_bb = MembershipInferenceBlackBox(target_model, attack_model_type='gb')
    # mlp_attack_bb.fit(x_train, y_train, x_test, y_test)
    mlp_attack_bb.fit(pred=predictions, y=y_train, test_pred=test_predictions, test_y=y_test, x=None, test_x=None)
    return mlp_attack_bb
