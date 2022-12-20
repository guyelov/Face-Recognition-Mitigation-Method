import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from tqdm import tqdm
from itertools import combinations

def load_CelebA(dir,property = 'None'):
    """
    Load the CelebA dataset.
    :param dir: Required. Type: str. The path where the CelebA images are.
    :param property: Required. Type: str. The property to use when splitting the data.
    :return: celeba_df: Type: pd.DataFrame. The CelebA dataset.
    """
    id_df = pd.read_csv(os.path.join(dir, '../../identity_CelebA.csv'), delimiter=' ', names=['path', 'id'])
    id_df['path'] = dir + id_df['path'].astype(str)
    if property == "None":
        id_df.dropna(inplace=True)
        return id_df
    path_df = pd.DataFrame(columns=['path'])
    property_df = pd.read_csv(os.path.join(dir, '../../list_attr_celeba.csv'), usecols=['image_id', property], header=0)
    property_df = property_df.rename(columns={'image_id': 'path'})
    property_df['path'] = dir + property_df['path'].astype(str)  # Concatenate dir path to every img name
    files = glob.glob(dir + '*.jpg') + glob.glob(dir + '*.JPG')
    path_df["path"] = files
    df = path_df.merge(id_df, on='path').merge(property_df, on='path')
    return df

def CelebA_split_ids_train_test(dir, property="None", seed=1):
    """
    Split the CelebA dataset into train and test sets based on the identity of the person in the image.
    :param dir: Required. Type: str. The path where the CelebA images are.
    :param property: Required. Type: str. The property to use when splitting the data.
    :param seed: Required. Type: int. The seed to use when splitting the data.
    :return: train_df: Type: pd.DataFrame. The train set.
    :return: test_df: Type: pd.DataFrame. The test set.
    """

    celeba_df = load_CelebA(dir, property=property)
    if property == "None":
        ids_df = celeba_df.drop_duplicates(subset=['id'])
        model_train_ids, attack_ids = train_test_split(ids_df, test_size=0.5, random_state=seed)
        attack_train_ids, attack_test_ids = train_test_split(attack_ids, test_size=0.5,
                                                             random_state=seed)
        return model_train_ids['id'].to_numpy(), attack_train_ids['id'].to_numpy(), attack_test_ids[
            'id'].to_numpy(), celeba_df
    else:
        ids_property_df = celeba_df.drop_duplicates(subset='id')[['id', property]]
        positive_records = ids_property_df[ids_property_df[property] == 1]
        negative_records = ids_property_df[ids_property_df[property] == -1]
        minority_attr = 1 if len(positive_records) < len(negative_records) else -1
        minority_ids = ids_property_df[ids_property_df[property] == minority_attr]
        majority_ids = ids_property_df[ids_property_df[property] == -minority_attr]
        minority_model_train_ids, minority_attack_ids = train_test_split(minority_ids, test_size=0.5, random_state=seed)
        minority_attack_train_ids, minority_attack_test_ids = train_test_split(minority_attack_ids, test_size=0.5,
                                                                               random_state=seed)
        majority_model_train_ids, majority_attack_ids = train_test_split(majority_ids,
                                                                         test_size=len(minority_attack_ids),
                                                                         random_state=seed)  #####
        majority_attack_train_ids, majority_attack_test_ids = train_test_split(majority_attack_ids, test_size=0.5,
                                                                               random_state=seed)
        model_train_ids = np.concatenate((minority_model_train_ids['id'].values, majority_model_train_ids['id'].values))
        attack_train_ids = np.concatenate(
            (minority_attack_train_ids['id'].values, majority_attack_train_ids['id'].values))
        attack_test_ids = np.concatenate((minority_attack_test_ids['id'].values, majority_attack_test_ids['id'].values))
        return model_train_ids, attack_train_ids, attack_test_ids, celeba_df

def CelebA_create_yes_records(data, save_to=None):
    """
    Create data of pairs of images from the same person.
    Currently, this function supposes to be applied on the CelebA dataset.
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id"].
    :param save_to: Optional. Type: str. The saving path.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    clip_dfs = []
    ids = data["id"].unique()
    for id in ids:
        data_id = data[data["id"] == id]
        data_id = data_id[:15]
        clip_dfs.append(data_id)
    data = pd.concat(clip_dfs, ignore_index=True)
    df = data.groupby('id')['path'].apply(combinations, 2) \
        .apply(list).apply(pd.Series) \
        .stack().apply(pd.Series) \
        .set_axis(['path1', 'path2'], 1, inplace=False) \
        .reset_index(level=0)
    df = df.drop('id', axis=1)
    df['label'] = 1
    if save_to is not None:
        df.to_csv(f'{save_to}celeba_positive_paired_data.csv', index=False)
    return df

def CelebA_create_no_records(data, yes_pairs_path, save_to=None, seed=0):
    """
    Create data of pairs of images from the same person.
    Currently, this function supposes to be applied on the CelebA dataset.
    :param data: Required. Type: Dataframe. The dataframe with columns ["path", "id"].
    :param yes_pairs_path: Required. Type: str. The path of the "celeba_positive_paired_data" file.
    :param save_to: Optional. Type: str. The saving path.
    :param seed: Optional. Type: int. The seed to use.
    :return: DataFrame that contains the columns ["path1","path2","label"].
    """
    random.seed(seed)

    pairs = pd.read_csv("{}celeba_positive_paired_data.csv".format(yes_pairs_path))
    pairs["path"] = pairs["path1"]
    pairs = pairs.join(data.set_index("path"), on="path")
    pairs = pairs[["path1", "path2", "label", "id"]]
    path2 = []
    ids = list(pairs["id"].unique())
    ids_options = dict()
    for id in tqdm(ids):
        ids_options.update({id: list(pairs[pairs["id"] != id]["path2"].unique())})
    for i, row in tqdm(pairs.iterrows()):
        id = row["id"]
        options = ids_options.get(id)
        new_path2 = random.sample(options, 1)[0]
        path2.append(new_path2)
        options.remove(new_path2)
        ids_options.update({id: options})
    pairs["path2"] = path2
    pairs = pairs.drop(["id"], axis=1)
    pairs["label"] = 0
    if save_to is not None:
        pairs.to_csv(f'{save_to}celeba_negative_paired_data.csv', index=False)
    return pairs
