import os
import pandas as pd
import numpy as np
import torch
from PIL import Image

class DrinksDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms):
        self.root = root
        self.transforms = transforms

        # Read csv file
        self.df = pd.read_csv(csv_file)

        # Manipulate dataframe format
        self.df = self.df[['frame','xmin','ymin','xmax','ymax','class_id']]
        self.df['area'] = (self.df['xmax'] - self.df['xmin']) * (self.df['ymax'] - self.df['ymin']) # Add column for area of bounding box

        # img filenames
        self.imgs = np.unique(self.df.iloc[:, 0].values)
        
        self.bbox_dict = {}
        self.labels_dict = {}
        self.area_dict = {}
        for r in self.imgs:
            self.bbox_dict[r] = []
            self.labels_dict[r] = []
            self.area_dict[r] = []

        df_values = self.df.values
        
        # create dictionaries for each parameter
 
        for d in df_values:
            # bbox
            val_bbox = d[1:5].astype(np.float32)
            self.bbox_dict[d[0]].append(val_bbox)

            # labels
            val_labels = np.array(d[5], dtype=np.int64)
            self.labels_dict[d[0]].append(val_labels)

            # area
            val_area = np.array(d[6], dtype=np.float32)
            self.area_dict[d[0]].append(val_area)

        # convert everything to tensors and combine dictionaries into one

        self.dictionary = {}
        idx = 0
        for k in self.bbox_dict:
            self.bbox_dict[k] = torch.from_numpy(np.array(self.bbox_dict[k]))
            self.labels_dict[k] = torch.from_numpy(np.array(self.labels_dict[k]))
            self.area_dict[k] = torch.from_numpy(np.array(self.area_dict[k]))

            self.dictionary[k] = {
                'boxes': self.bbox_dict[k],
                'labels': self.labels_dict[k],
                'image_id': torch.tensor([idx], dtype=torch.int64, device='cpu'),
                'area': self.area_dict[k],
                'iscrowd': torch.zeros(self.bbox_dict[k].shape[0], dtype=torch.uint8, device='cpu')
            }
            idx += 1

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]

        img = Image.open(os.path.join(self.root, img_name))

        boxes = self.dictionary[img_name]['boxes']
        labels = self.dictionary[img_name]['labels']
        image_id = self.dictionary[img_name]['image_id']
        area = self.dictionary[img_name]['area']
        iscrowd = self.dictionary[img_name]['iscrowd']

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target