# pytorch-VideoDataloader
Tools for loading video dataset and transforms on video in pytorch. You can directly load video files without preprocessing.

## Requirements

+ pytorch
+ torchvision
+ numpy
+ python-opencv
+ PIL

## How to use

1. Place the files [datasets.py](./datasets.py) and [transforms.py](./transforms.py) at your project directory.

2. Arrange the videos into class-wise folder

   ```csv
   path
   ~/path/class1/file1.mp4
   ~/path/class1/file2.mp4
   ~/path/class2/file3.mp4
   ~/path/class2/file4.mp4
   ```

# Loading the dataset
### **datasets.VideoDataloader**
If you have different training and testing folder in the dataset, use this one to load train and test data separately.

   ```python
   import torch
   import torchvision
   import datasets
   import transforms
   
   train_dataset = datasets.VideoDataloader('C:\\Users\\shuvendu\\Desktop\\UCF-101\\train',
             transform=torchvision.transforms.Compose([
                 transforms.VideoToTensor(max_len=16),
                 transforms.VideoRandomCrop([236, 236]),
                 transforms.VideoResize([224, 224]),
                 transforms.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                                            std=[0.22803, 0.22145, 0.216989]),
             ])
        )

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 2, shuffle = True)
   for videos in train_loader:
       print(videos.size())
   ```

### **datasets.VideoFolderDataloader**
If all videos are in same folder and you need to split it into train and test folders.

   ```python
   import torch
   import torchvision
   import datasets
   import transforms
   
    dl = datasets.VideoFolderDataloader('C:\\Users\\shuvendu\\Desktop\\UCF',
           train_ratio=0.9,

           train_transform=torchvision.transforms.Compose([
               transforms.VideoToTensor(max_len=16),
               transforms.VideoRandomCrop([236, 236]),
               transforms.VideoResize([224, 224]),
               transforms.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                                        std=[0.22803, 0.22145, 0.216989]),
           ]),

           test_transform=torchvision.transforms.Compose([
               transforms.VideoToTensor(max_len=16),
               transforms.VideoCenterCrop([236, 236]),
               transforms.VideoResize([224, 224]),
               transforms.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                                    std=[0.22803, 0.22145, 0.216989]),
           ])
        )
    
    train_loader = torch.utils.data.DataLoader(dl.train_dataset, batch_size = 2, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dl.test_dataset, batch_size = 2, shuffle = True)
    
for videos in train_loader:
       print(videos.size())
   ```
## Docs

### [Transforms](./transforms.py)

+ #### **transforms.VideoToTensor** 
Best frame sample to cover full video
load video at given file path to torch.Tensor (C x L x H x W, C = 3) 
It can be composed with torchvision.transforms.Compose().

  + **Parameters**
    + **max_len (int)**: Maximum output time depth (L <= max_len). Default is None.
        If it is set to None, it will output all frames.
    + **fps (int)**: sample frame per seconds. It must lower than or equal the origin video fps.
        Default is None.
    + **padding_mode (str)**: Type of padding. Default to None. Only available when max_len is not None.
        - None: won't padding, video length is variable.
        - 'zero': padding the rest empty frames to zeros.
        - 'last': padding the rest empty frames to the last frame.

+ #### **transforms.VideoFilePathToTensor** 
  load video at given file path to torch.Tensor (C x L x H x W, C = 3). 



  + **Parameters**
    + **max_len** (int): Maximum output time depth (L <= max_len). Default is None. If it is set to None, it will output all frames. 
    + **fps** (int): sample frame per seconds. It must lower than or equal the origin video fps. Defaults to None. 
    + **padding_mode** (str): Type of padding. Default to None. Only available when max_len is not None.
      + None: won't padding, video length is variable.
      + 'zero': padding the rest empty frames to zeros.
      + 'last': padding the rest empty frames to the last frame.

+ #### **transforms.VideoFolderPathToTensor**

  load video at given folder path to torch.Tensor (C x L x H x W).

  + **Parameters**
    + **max_len** (int): Maximum output time depth (L <= max_len). Default is None. If it is set to None, it will output all frames. 
    + **padding_mode** (str): Type of padding. Default to None. Only available when max_len is not None.
      + None: won't padding, video length is variable.
      + 'zero': padding the rest empty frames to zeros.
      + 'last': padding the rest empty frames to the last frame.

+ #### **transforms.VideoResize**

  resize video tensor (C x L x H x W) to (C x L x h x w).

  + **Parameters**
    + **size** (sequence): Desired output size. size is a sequence like (H, W), output size will matched to this.
    + **interpolation** (int, optional): Desired interpolation. Default is `PIL.Image.BILINEAR`

+ #### **transforms.VideoRandomCrop**

  Crop the given Video Tensor (C x L x H x W) at a random location.

  + **Parameters**
    + **size** (sequence): Desired output size like (h, w).

+ #### **transforms.VideoCenterCrop**

  Crops the given video tensor (C x L x H x W) at the center.

  + **Parameters**
    + **size** (sequence): Desired output size of the crop like (h, w).

+ #### **transforms.VideoRandomHorizontalFlip**

  Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

  + **Parameters**
    + **p** (float): probability of the video being flipped. Default value is 0.5.

+ #### **transforms.VideoRandomVerticalFlip**

  Vertical flip the given video tensor (C x L x H x W) randomly with a given probability.

  + **Parameters**
    + **p** (float): probability of the video being flipped. Default value is 0.5.

+ #### **transforms.VideoGrayscale**

  Convert video (C x L x H x W) to grayscale (C' x L x H x W, C' = 1 or 3)

  + **Parameters**
    + **num_output_channels** (int): (1 or 3) number of channels desired for output video.
