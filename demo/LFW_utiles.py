import os
import shutil

# from facenet_pytorch import MTCNN
import numpy as np
from tqdm import tqdm
import pandas as pd

lfw_path = 'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\LFW_Demo'
def load_lfw(attack=False):
    """"""
    lfw_train = pd.read_csv(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\demo_train.csv')
    lfw_test = pd.read_csv(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\demo_test.csv')
    train_images = lfw_train['image1'].tolist() + lfw_train['image2'].tolist()
    test_images = lfw_test['image1'].tolist() + lfw_test['image2'].tolist()
    if attack:
        lfw_attack = pd.read_csv(
            'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\attack_demo_data.csv')
        attack_images = lfw_attack['image1'].tolist() + lfw_attack['image2'].tolist()
        return train_images, attack_images
    return train_images, test_images

def create_membership_pairs(train_set,attacker_set):
    names_train = []

    for index, row in train_set.iterrows():
        name1 = row['image1'].split('\\')[-1]  # split the file path by '' and then by '', and take all elements except the last (4-digit number)
        name2 = row['image2'].split('\\')[-1] # same process for image2

        full_name1 = ''.join(name1)  # join first and last name with a space
        full_name2 = ''.join(name2)  # same for image2
        names_train.append(full_name1 + ' and ' + full_name2)  # combine the two names and append to the list
    names_attacker = []
    for index, row in attacker_set.iterrows():
        name1 = row['image1'].split('\\')[-1]
        name2 = row['image2'].split('\\')[-1]
        full_name1 = ''.join(name1)
        full_name2 = ''.join(name2)
        names_attacker.append(full_name1 + ' and ' + full_name2)
    names_train = np.array(names_train)
    names_attacker = np.array(names_attacker)
    np.save('C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\names_train.npy', names_train)
    np.save('C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\names_attacker.npy', names_attacker)
    return names_train, names_attacker
def create_lfw_folder():
    root_folder  = "C:\\Users\guyel\PycharmProjects\Deep Learning\Assigment 2\lfw-py rgb\lfw_funneled"
    destination_folder  = "C:\\Users\guyel\PycharmProjects\Face Recognition Mitigation Method\demo\Data\LFW_Demo"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over the subfolders
    total_images = sum([len(files) for r, d, files in os.walk(root_folder)])

    with tqdm(total=total_images) as pbar:
        for subfolder in os.listdir(root_folder):
            subfolder_path = os.path.join(root_folder, subfolder)
            if os.path.isdir(subfolder_path):
                # Iterate over the images in the subfolder
                for image in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, image)
                    # Check if the file is an image
                    if image.endswith('.jpg') or image.endswith('.png'):
                        # Copy the image to the destination folder
                        shutil.copy(image_path, destination_folder)
                        # Update the progress bar
                        pbar.update(1)

    print('Done!')
def num_intersection(lst1, lst2):
    return len(list(set(lst1) & set(lst2)))
    print('Done!')
if __name__ == '__main__':
    train_set = pd.read_csv(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\target_model_demo_data.csv')
    attacker_set = pd.read_csv(
        'C:\\Users\\guyel\\PycharmProjects\\Face Recognition Mitigation Method\\demo\\Data\\attack_demo_data.csv')
    # create_lfw_folder()
    names_train, names_attacker = create_membership_pairs(train_set, attacker_set)
    print(num_intersection(names_train, names_attacker))
    print(len(names_train))
    print(len(names_attacker))
