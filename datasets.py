import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import os
import torchvision
import transforms

__all__ = ['VideoFolderDataloader', 'VideoFileDataloader', 'VideoDataloader']


class VideoFolderDataloader(object):
    def __init__(self, folder_location, train_ratio=0.9, train_transform=None, test_transform=None):
        self.folder_location = folder_location
        self.train_ratio = train_ratio

        self.all_videos = []
        for path, subdirs, files in os.walk(self.folder_location):
            for name in files:
                self.all_videos.append(os.path.join(path, name))

        np.random.permutation(range(len(self.all_videos)))
        train_files = self.all_videos[:int(len(self.all_videos) * self.train_ratio)]
        test_files = self.all_videos[int(len(self.all_videos) * self.train_ratio):]

        self.train_dataset = VideoFileDataloader(train_files, True, train_transform)
        self.test_dataset = VideoFileDataloader(test_files, False, test_transform)


class VideoFileDataloader(Dataset):
    def __init__(self, video_list, train=False, transform=None):
        self.transform = transform
        self.train = train
        self.all_videos = video_list

        self.seperator = '/'
        if '\\' in self.all_videos[0]:
            self.seperator = '\\'

        self.all_classes = []
        for file in self.all_videos:
            self.all_classes.append(file.split(self.seperator)[-2])

        class_dict = {}
        for i, c in enumerate(set(self.all_classes)):
            class_dict[c] = i

        self.labels = []
        for c in self.all_classes:
            self.labels.append(class_dict[c])

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, index):
        video = self.all_videos[index]
        label = self.labels[index]
        if self.transform:
            video = self.transform(video)
        return video, label


class VideoDataloader(Dataset):
    def __init__(self, folder_location, transform=None):
        self.folder_location = folder_location
        self.transform = transform

        self.all_videos = []
        for path, subdirs, files in os.walk(self.folder_location):
            for name in files:
                self.all_videos.append(os.path.join(path, name))

        self.seperator = '/'
        if '\\' in self.all_videos[0]:
            self.seperator = '\\'

        self.all_classes = []
        for file in self.all_videos:
            self.all_classes.append(file.split(self.seperator)[-2])

        class_dict = {}
        for i, c in enumerate(set(self.all_classes)):
            class_dict[c] = i

        self.labels = []
        for c in self.all_classes:
            self.labels.append(class_dict[c])

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, index):
        video = self.all_videos[index]
        label = self.labels[index]
        if self.transform:
            video = self.transform(video)
        return video, label


if __name__ == '__main__':
    dl = VideoDataloader('D:\\Dataset\\Video\\UCF\\UCF-101',
                         transform=torchvision.transforms.Compose([
                             transforms.TorchVideoToTensor(max_len=16),
                             transforms.VideoRandomCrop([236, 236]),
                             transforms.VideoResize([224, 224]),
                         ])
                         )
    video, label = dl[0]

    dl = VideoFolderDataloader('D:\\Dataset\\Video\\UCF\\UCF-101',
           train_ratio=0.9,
           train_transform=torchvision.transforms.Compose([
               transforms.TorchVideoToTensor(max_len=16),
               transforms.VideoRandomCrop([236, 236]),
               transforms.VideoResize([224, 224]),
           ]),

           test_transform=torchvision.transforms.Compose([
               transforms.TorchVideoToTensor(max_len=16),
               transforms.VideoCenterCrop([236, 236]),
               transforms.VideoResize([224, 224]),
           ])
        )

    video, label = dl.train_dataset[0]

    frame1 = torchvision.transforms.ToPILImage()(video[:, 0, :, :])
    frame2 = torchvision.transforms.ToPILImage()(video[:, 15, :, :])
    frame1.show()
    frame2.show()

    test_loader = torch.utils.data.DataLoader(dl.test_dataset, batch_size=1, shuffle=True)

    for videos, labels in test_loader:
        print(videos.size(), label)