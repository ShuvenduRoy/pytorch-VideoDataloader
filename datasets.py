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

        self.seperator = '/'
        if '\\' in self.all_videos[0]:
            self.seperator = '\\'

        self.all_classes = []
        for file in self.all_videos:
            self.all_classes.append(file.split(self.seperator)[-2])

        self.class_names = list(set(self.all_classes))
        self.class_names.sort()

        self.class_dict = {}
        for i, c in enumerate(self.class_names):
            self.class_dict[c] = i
        print(self.class_dict)

        np.random.shuffle(self.all_videos)
        train_files = self.all_videos[:int(len(self.all_videos) * self.train_ratio)]
        test_files = self.all_videos[int(len(self.all_videos) * self.train_ratio):]
        print('Training examples: ', len(train_files))
        print('Testing examples: ', len(test_files))
        self.train_dl = VideoFileDataloader(train_files, train_transform, self.class_dict)
        self.test_dl = VideoFileDataloader(test_files, test_transform, self.class_dict)


class VideoFileDataloader(Dataset):
    def __init__(self, video_list, transform=None, class_dict=None):
        self.transform = transform
        self.all_videos = video_list
        self.class_dict = class_dict

        self.seperator = '/'
        if '\\' in self.all_videos[0]:
            self.seperator = '\\'

        self.all_classes = []
        for file in self.all_videos:
            self.all_classes.append(file.split(self.seperator)[-2])

        self.labels = []
        for c in self.all_classes:
            self.labels.append(self.class_dict[c])

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


class VideoDataloader(object):
    def __init__(self, train_folder_location, test_folder_location=None, train_transform=None, test_transform=None):
        self.train_folder_location = train_folder_location
        self.test_folder_location = test_folder_location
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.train_videos = []
        for path, subdirs, files in os.walk(self.train_folder_location):
            for name in files:
                self.train_videos.append(os.path.join(path, name))
        print('Total number of training datapoints ', len(self.train_videos))

        self.test_videos = []
        for path, subdirs, files in os.walk(self.test_folder_location):
            for name in files:
                self.test_videos.append(os.path.join(path, name))
        print('Total number of testing datapoints ', len(self.test_videos))

        self.seperator = '/'
        if '\\' in self.train_videos[0]:
            self.seperator = '\\'

        self.all_classes = []
        for file in self.train_videos:
            self.all_classes.append(file.split(self.seperator)[-2])

        self.class_names = list(set(self.all_classes))
        self.class_names.sort()

        self.class_dict = {}
        for i, c in enumerate(self.class_names):
            self.class_dict[c] = i
        print(self.class_dict)

        self.train_dl = VideoFileDataloader(self.train_videos, train_transform, self.class_dict)
        self.test_dl = VideoFileDataloader(self.test_videos, test_transform, self.class_dict)


if __name__ == '__main__':
    import transforms
    import torchvision
    import torch

    train_loader = VideoFolderDataloader('D:\\Dataset\\Video\\UCF\\UCF-101',
                                                  train_ratio=0.9,
                                                  train_transform=torchvision.transforms.Compose([
                                                      transforms.TorchVideoToTensor(max_len=16),
                                                      transforms.VideoRandomCrop([236, 236]),
                                                      transforms.VideoResize([224,224]),
                                                  ]),

                                                  test_transform=torchvision.transforms.Compose([
                                                      transforms.TorchVideoToTensor(max_len=16),
                                                      transforms.VideoCenterCrop([236, 236]),
                                                      transforms.VideoResize([224,224]),
                                                  ])
                                                  )
    video, label = train_loader.train_dl[0]
    train_loader.train_dl = torch.utils.data.DataLoader(train_loader.train_dl, batch_size=1,
                                                         shuffle=True)
    train_loader.test_dl = torch.utils.data.DataLoader(train_loader.test_dl, 1,
                                                       shuffle=True)
    # this is just to keep the coding part same in training loop
    test_loader = train_loader

    frame1 = torchvision.transforms.ToPILImage()(video[:, 0, :, :])
    frame2 = torchvision.transforms.ToPILImage()(video[:, 15, :, :])
    frame1.show()
    frame2.show()

    video, label = next(iter(train_loader.train_dl))
    print(video.shape)
    print(label.shape)

    # all_cls = set()
    # for video, label in train_loader.test_dl:
    #     all_cls.add(int(label))
    # print(len(all_cls))
    # print(all_cls)
