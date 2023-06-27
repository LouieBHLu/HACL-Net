import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class MIL_dataloader():
    def __init__(self, data_path, label_train_val=None, train=True):
        if train:

            X_train, X_test = train_test_split(data_path, test_size=0.1, random_state=0, stratify=label_train_val)  # 10% validation
 
            traindataset = MIL_dataset(path=X_train)

            traindataloader = DataLoader(traindataset,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=4)

            valdataset = MIL_dataset(path=X_test)

            valdataloader = DataLoader(valdataset,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=4) 

            self.dataloader = [traindataloader, valdataloader]

        else:

            testdataset = MIL_dataset(path=data_path)

            testloader = DataLoader(testdataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=4)

            self.dataloader = testloader

    def get_loader(self):
        return self.dataloader


class MIL_dataset(Dataset):
    def __init__(self, path):

        self.path = path    

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        img_path = self.path[idx]

        vgg_file = np.load(img_path)
        cur_vgg = vgg_file['vgg_features']
        cur_label = vgg_file['label']
        if int(cur_label) == 3 or int(cur_label) == 2: 
            cur_label = 1

        if cur_vgg.shape[0] == 0: raise 0

        label_tensor = torch.tensor([1]) if cur_label == 1 else torch.tensor([0])
        sample = {'feat': torch.from_numpy(cur_vgg), 'label': label_tensor}

        return sample
