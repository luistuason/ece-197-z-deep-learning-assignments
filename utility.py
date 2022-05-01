import numpy as np
import os
import pandas as pd
from typing import Union, List, Any, Optional
import torch

from dataset import *
import lib.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

cwd = os.path.abspath(os.path.dirname(__file__))
labels_train_path = os.path.abspath(os.path.join(cwd, 'data/drinks/labels_train.csv'))
labels_test_path = os.path.abspath(os.path.join(cwd, 'data/drinks/labels_test.csv'))

df_train = pd.read_csv(labels_train_path)
df_test = pd.read_csv(labels_test_path)


drinks_dataset = DrinksDataset(root='data/drinks/', csv_file=labels_train_path, transforms=get_transform(train=True))



print(drinks_dataset.imgs)