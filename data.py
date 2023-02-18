import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from preprocess_data import create_dir

class PatchLoader(Dataset): 
    ''' Loads only the patches specified in csv. '''

    def __init__(self, csv_path, data_dir, mode, res_root_dir): 

        self.data_dir = data_dir
        self.mode = mode 
        # self.fold_idx = fold_idx
        self.res_root_dir = res_root_dir
        self.res_dir = os.path.join(self.res_root_dir, 'dataset', 'train')
        create_dir(self.res_dir)

        self.csv_path = csv_path
        self.csv_data = pd.read_csv(self.csv_path)


    def __len__(self):
        if self.mode == 'train': 
            return len(self.csv_data)
        elif self.mode == 'test': 
            # return all patches in image
            pass

    def __getitem__(self, idx): 

        im_path = os.path.join(self.data_dir, self.csv_data['name'][idx])
        name = os.path.basename(im_path).split('.')[0]
        im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        lbl = self.csv_data['class'][idx]

        y_min, y_max = int(self.csv_data['y_min'][idx]), int(self.csv_data['y_max'][idx])
        x_min, x_max = int(self.csv_data['x_min'][idx]), int(self.csv_data['x_max'][idx])
        patch = im[y_min:y_max, x_min:x_max]

        # save dataset 
        save_path = os.path.join(self.res_dir, os.path.basename(im_path))
        cv2.imwrite(save_path, patch)
     
        # (channels, y, x)
        patch = np.moveaxis(patch, -1, 0)

        # convert to tensor
        patch = torch.from_numpy(patch).type(torch.float32)/255.

        return patch, lbl, name
