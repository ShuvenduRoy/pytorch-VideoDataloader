from torch.utils.data import Dataset
import numpy as np
import os

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
        print('Training examples: ', len(train_files))
        print('Testing examples: ', len(test_files))
        self.train_dl = VideoFileDataloader(train_files, True, train_transform)
        self.test_dl = VideoFileDataloader(test_files, False, test_transform)


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
        try:
            video = self.all_videos[index]
            label = self.labels[index]
            if self.transform:
                video = self.transform(video)

        except Exception as e:
            print(e)
            print('Error with file: ', self.all_videos[index])

            video = self.all_videos[index+1]
            label = self.labels[index+1]
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
        print('Total number of datapoints ', len(self.all_videos))

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

