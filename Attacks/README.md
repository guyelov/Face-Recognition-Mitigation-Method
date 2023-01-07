# Attacks :no_entry:

This folder contains existing privacy violation attacks that can be applied to face recognition (FR) systems.
These attacks can be used to test the effectiveness of the FR system and the mitigation method implemented in
this repository.
The implementation of the attacks is based one the [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
(ART) library.

Currently, the following attacks are supported:
- [Membership Inference](https://github.com/guyelov/Face-Recognition-Mitigation-Method/blob/38cd300509632d4f87279188deb305ceedf2a48b/Attacks/Membership_Inference/MembershipAttack.py): This attack attempts to determine whether an individual's data was used to train the FR model.

- In the future we plan to add support for the following attacks:
- Model Inversion: :construction: This attack attempts to reconstruct the FR model from the data used to train it.
- Model Extraction: :ninja: This attack attempts to extract or steal the FR model from the system.
- Attribute Inference: :woman: This attack attempts to determine the attributes of an individual from the FR model.
- External information leakage: :mag: This attack attempts to determine the attributes of an individual from external information.

Please note that these attacks are for research and testing purposes only and should not be used maliciously. :warning: