from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch
import sys
import torchvision.transforms as T

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


import torch.utils.data as data
class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, __C, split, transform=None, target_transform=None):

        self.__C = __C
        self.root = __C.data_dir
        self.train_list = __C.train_list
        self.valid_list = __C.valid_list
        self.test_list = __C.test_list
        self.split = split  # training set or test set
        # self.transform = lambda x: (torch.from_numpy(x.transpose((2, 0, 1))).float().div(255) - 0.5).div(0.5)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.target_transform = target_transform
        self.max_cls_class = __C.max_cls_class
        self.max_per_class = __C.max_per_class

        if self.split == 'train':
            data_list = self.train_list
        elif self.split == 'valid':
            data_list = self.valid_list
        elif self.split == 'test':
            data_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.limit_class_and_examples()


    def limit_class_and_examples(self):
        self.avaialbe_indexes = []
        self.class_dict = {}
        for i,target in enumerate(self.targets):
            if target not in self.class_dict:   # limit class number
                if len(self.class_dict)<self.max_cls_class:
                    self.class_dict[target]=[i]
                    self.avaialbe_indexes.append(i)
            elif len(self.class_dict[target])<self.max_per_class:   # limit examples per class
                self.class_dict[target].append(i)
                self.avaialbe_indexes.append(i)


        self._load_meta()
        print(self.classes)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = self.avaialbe_indexes[index]
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        item = {'data':img, 'class_id': target}
        return item


    def __len__(self):
        return len(self.avaialbe_indexes)



def get_dataloader(__C, split):
    dataset = CIFAR10(__C, split)
    data_loader = data.DataLoader(dataset,
        batch_size=__C.batch_size,
        num_workers=__C.num_workers,
        shuffle= split=="train",
        # pin_memory=__C.pin_memory
        )
    return data_loader


