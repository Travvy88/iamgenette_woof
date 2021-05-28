import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader


class WoofDataset(Dataset):
    def __init__(self, folder, img_size = (256, 256), aug=None):
        self.csv, self.class_dict = self.create_csv(folder)
        self.X = self.csv.values[:, 0]
        self.y = self.csv.values[:, 1]
        
        self.img_size = img_size
        self.aug = aug
        
        
        
    def create_csv(self, data_dir):
        """
        Вход: директория с датасетом 
        Выход: csv таблица с путями файлов и классами
        """
        classes = []
        paths = []

        k = 0
        for label in os.listdir(data_dir):
            for photo in os.listdir(os.path.join(data_dir, label)):
                classes.append(label)
                paths.append(os.path.join(data_dir, label, photo))


        data = pd.DataFrame() 

        people = pd.Series(classes, name='class')
        paths = pd.Series(paths, name ='path')
        data = pd.concat((paths, people), axis=1)
        # data.to_csv(csv_name, index=False)  # сохраняем таблицу

        classes = np.unique(data.values[:, 1])
        numbers = np.arange(0, len(classes))
        class_dict = dict(zip(classes, numbers))
        
        classes_nums = [class_dict[k] for k in data['class'].values]
        data['class'] = classes_nums
        
        return data, class_dict

    def __len__(self):
        return (len(self.X))            
    
    def get_image(self, path):
        img = Image.open(path).resize(self.img_size)
        if len(np.array(img).shape) == 2:  # если grayscale то в RGB
            img = img.convert('RGB')
        img = np.array(img)
        if self.aug:
            img = self.aug(image=img)['image']            
        
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return img / 255
    
    def get_PIL(self, i, resize=True):
        if resize:
            return Image.open(self.X[i]).resize(self.img_size).convert('RGB')
        else:
            return Image.open(self.X[i]).convert('RGB')
    
    def show(self, i):
        img = Image.open(self.X[i])
        plt.imshow(img)
        plt.show()
        
    def __getitem__(self, i):
        return self.get_image(self.X[i]), self.y[i]
    