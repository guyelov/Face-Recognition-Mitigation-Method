# Face Recognition System Structure:
<p align="center">
  <img alt="example image" src="https://github.com/guyelov/Face-Recognition-Mitigation-Method/blob/3fb8623bc12869e43a3264788a76da674288a8ea/Data/Images/FR%20System%20Structure.jpg" width="700"/>
</p>

The FR system consists of three main components:
- **Embedder**: The embedder is a neural network that takes an image as input and outputs a 512-dimensional vector. The embedder is trained to learn a mapping from an image to a vector that represents the image. The embedder is trained on the CelebA dataset.
For the demo, there is only one backbone model for the embedder. However, the embedder can be trained with different backbone models (such as ResNet50, ResNet101, etc.) the backbone of the demo embedder is IResNet100.
- **Predictor**: The predictor is a neural network that takes the output of the embedder as input and outputs a 1-dimensional vector. The predictor is trained to learn whether two embeddings belong to the same person or not. The predictor presented in the demo is trained on the LFW dataset.
- **FR System**: The FR system is the combination of the embedder and predictor. The FR system takes an image as input and outputs a 1-dimensional vector that represents if the image belongs to the same person or not.
The model is trained to classify whether two images are of the same person or not.
