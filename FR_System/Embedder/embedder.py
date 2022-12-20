from FR_System.Embedder import iresnet as AFBackbone
import torch
from ModelX_Zoo.test_protocol.utils.model_loader import ModelLoader
from ModelX_Zoo.backbone.backbone_def import BackboneFactory


class Embedder(torch.nn.Module):
    """
    The Embedder class, the embedder is a pretrained model that creates embedding vectors from images.
    """

    def __init__(self, device, model_name='', train=False, faceX_zoo=True):
        """
        The Embedder class constructor. The constructor loads the model and sets the device.
        :param device: Required. Type: str. The device the network will use.
                       Options:"cpu" / "cuda:0" / other coda name. The device used by pytorch.
        :param model_name: Optional. Type: str. The name of the model to use.
        :param train: Optional. Type: bool. If True, the model will be in training mode.
        :param faceX_zoo: Optional. Type: bool. If True, the model will be loaded from FaceX-Zoo.

        """
        super(Embedder, self).__init__()
        self.embedder_name = model_name
        if not model_name[0].isupper():
            faceX_zoo = False
        if model_name == 'iresnet100':
            if train:
                embedder = AFBackbone.iresnet100(pretrained=True).to(device).train()

            else:
                embedder = AFBackbone.iresnet100(pretrained=True).to(device).eval()

        ######################################
        # FaceX-Zoo models
        ######################################
        if faceX_zoo:
            print(f"Model Name: {model_name}")
            model_path = f"/sise/home/guyelov/FR_System/Pretrained Backbones/{model_name}.pt"
            conf_path = r"/sise/home/guyelov/FR_System/ModelX_Zoo/test_protocol/backbone_conf.yaml"
            ModelFactory = BackboneFactory(model_name, conf_path)
            model_loader = ModelLoader(ModelFactory)
            if train:
                # set the model to train mode
                embedder = model_loader.load_model(model_path=model_path, device=device, train_mode=True)
            else:
                embedder = model_loader.load_model(model_path=model_path, device=device, train_mode=False)

        self.embedder = embedder
        self.device = device

    def forward_ones(self, input):
        """
        The method returns the input's embedding vectors produced by the embedder.
        :param input: Required. Type: ndarray / torch.tensor. An array like object with 4 dimensions:
                        (batch size, channels, image height, image width).
                        For example: (24, 3, 112, 112).
        :return: torch.tensor. The embedding vectors for the inputs.
                    Shape: (batch size, embedding vector size).
                    For example: (24, 512)
        """
        if torch.is_tensor(input):
            return self.embedder(input.float()).to(self.device)
        else:
            inp = torch.tensor(input).float().to(self.device)
            return inp

    def forward(self, image_1, image_2):
        image_1_embed = self.forward_ones(image_1)
        image_2_embed = self.forward_ones(image_2)
        images_embed = torch.subtract(image_1_embed, image_2_embed)
        return images_embed.cpu().detach().numpy()


if __name__ == '__main__':
    pass
