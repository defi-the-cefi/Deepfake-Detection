### Overview ###
# Creating Pandas Dataframe with Image File Paths and True Class (can include annotation here)
# create pandas frame composed of list of all real images
# add column with [1,0] vector
# create second frame with fake data
# add column with [0,1] vector
# concatenate both frames
# create torch.dataset object using the dataframe we just created

#%%
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from torchvision.io import encode_jpeg
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

#%%
# image file paths
real_images_path = os.path.join('dataset', 'real')
fake_images_path = os.path.join('dataset', 'fake')

# list of all files in above paths
real_images_list = os.listdir(real_images_path)
fake_images_list = os.listdir(fake_images_path)
#turn into dataframe
real_frame = pd.DataFrame(real_images_list, columns=['file_name'])
print(real_frame)
fake_frame = pd.DataFrame(fake_images_list, columns=['file_name'])
print(fake_frame)

#%% creating pandas DataFrame for filepaths and annotations to sample from
# frame for real data
real_frame['dir_path'] = real_images_path
real_frame['file_path'] = real_frame.apply(lambda x: os.path.join(x['dir_path'],x['file_name']), axis=1)
real_frame['class'] = real_frame.apply(lambda x: [1,0], axis = 1)
print(real_frame)

#frame for fake data
fake_frame['dir_path'] = fake_images_path
fake_frame['file_path'] = fake_frame.apply(lambda x: os.path.join(x['dir_path'],x['file_name']), axis=1)
fake_frame['class'] = fake_frame.apply(lambda x: [0,1], axis = 1)
print(fake_frame)

# concatenate both frames
all_images_frame = pd.concat([real_frame,fake_frame], axis=0, ignore_index=True)
print(all_images_frame)

#%% create train, val, test sets
train_frame, test_frame_placeholder = train_test_split(all_images_frame, test_size=.3, stratify=all_images_frame['class'])
val_frame, test_frame = train_test_split(test_frame_placeholder, test_size=.5, stratify=test_frame_placeholder['class'])
print('train frame size: ', len(train_frame))
print('validation frame size: ',len(val_frame))


#%% Create our torch.Dataset class instance
class images_dataset(Dataset):
    """laod image dataset."""
    def __init__(self, dataset_spec_frame, min_image_size= 600, transform=True):
        """
            dataset_spec_frame (pd.DataFrame): Frame with image path and metadata.
            transform (bool): applies image transformations
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.dataset_details = dataset_spec_frame.reset_index(drop=True)
        self.min_image_size = min_image_size

    def __len__(self):
        return len(self.dataset_details)

    def __getitem__(self, index):
        binary_class_vec = torch.Tensor(self.dataset_details.loc[index,'class'])
        img_file_path = self.dataset_details.loc[index,'file_path']
        print(img_file_path)
        image = Image.open(img_file_path)

        trans_to_apply = transforms.Compose(
            [transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), # https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#autoaugment
             transforms.Resize(self.min_image_size,antialias=True),  # for training use transforms.RandomResizedCrop(min_image_size)
             transforms.ToTensor(),  # automagically converts all images to [0,1] values
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])  # standard procedure for image processing)
        if self.transform == True:
            sample = trans_to_apply(image)
            print(sample.shape)
        else:
            sample = image

        return sample, binary_class_vec

#%% test run above
# train_images = images_dataset(dataset_spec_frame=train_frame)
# train_loader = DataLoader(train_images, batch_size=1, shuffle=True)
# print(train_images[0])





