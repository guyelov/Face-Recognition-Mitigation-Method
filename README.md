# FaceBone - A Multiple Backbones Mitigation Method :lock:
In this repository, we have implemented a method to protect against privacy violation attacks on face recognition (FR) systems. The repository contains the FR system that the mitigation method will be applied to, as well as support for existing publicly available privacy violation attacks (such as membership inference and model inversion).

For more information about the method, please see the [system presentation](https://github.com/guyelov/Face-Recognition-Mitigation-Method/blob/master/demo/%5B01%5D%20FR%20Mitigation%20Methods%20Hackathon%20Demo.pdf).

In addition, we have included a demo app that allows users to test the model with their own images or images from the LFW (Labeled Faces in the Wild) dataset. The demo app demonstrates the performance of the FR system and shows how the mitigation method can protect against a membership attack.

The repository is organized into the following folders:

- [Attacks](https://github.com/guyelov/Face-Recognition-Mitigation-Method/tree/master/Attacks) - contains existing privacy violation attacks :no_entry:

- [Data](https://github.com/guyelov/Face-Recognition-Mitigation-Method/tree/master/Data) - contains pre-trained models and data utilities for processing images :file_folder:
- [FR System](https://github.com/guyelov/Face-Recognition-Mitigation-Method/tree/master/FR_System) - contains the face recognition model with the embedder and predictor
:closed_lock_with_key:
- [Demo](https://github.com/guyelov/Face-Recognition-Mitigation-Method/tree/master/demo) - contains the streamlet app for testing the FR system and mitigation method 
:computer:

## How to create target models:question:
To create a target model without the mitigation method, do the following:
- Clone the repository or download the files
- Download from [here](https://drive.google.com/file/d/1LeTK9EjtA2HeLIg3UDvSbngIL-PLdAQM/view?usp=sharing) the IResNet100 backbone and place it in the Data folder.
- Change in the [iresnet](https://github.com/guyelov/Face-Recognition-Mitigation-Method/blob/master/FR_System/Embedder/iresnet.py) at line 248  the path of the iresnet100.pth file to the path of the downloaded file.
- Run the [target_model_creation.py](https://github.com/guyelov/Face-Recognition-Mitigation-Method/blob/master/demo/target_model_creation_demo.py) file and this will create the target model.
Note that there is no need to download the LFW dataset because in the target_model_creation.py its downloaded automatically.

For more information about the target model structure, please refer to the [FR System](https://github.com/guyelov/Face-Recognition-Mitigation-Method/tree/master/FR_System) folder.

