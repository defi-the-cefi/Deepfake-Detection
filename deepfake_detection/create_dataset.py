#%%
# Import opencv
import cv2
# Import operating sys
import os
# Import matplotlib
from matplotlib import pyplot as plt
import random

# ToDO
#   1. download Celeb-DF
#   2. Parse a few hundred video files and record a single frame
#   3. Preprocess and data-augment

# final destination
real_writepath = r'deepfake_detection\dataset\real'
fake_writepath = r'deepfake_detection\dataset\fake'
# parent direcotry for real data
real_path = r'deepfake_detection\DFMNIST+\real_dataset\selected_train'
fake_path = r'deepfake_detection\DFMNIST+\fake_dataset'

# sample random files from real directory
real_vids = os.listdir(real_path)

#%% list all forder and go one level down
fake_dirs = os.listdir(fake_path)
print(fake_dirs)
random_fake_subdir = fake_dirs[random.randint(0,len(fake_dirs)-1)]
random_subdir_files = os.listdir(os.path.join(fake_path,random_fake_subdir))
print(random_subdir_files)
print('above is list of all files in: ', random_fake_subdir)

#%%
def sample_frame(parent_dir, real=True):
    files_in_dir = os.listdir(parent_dir)
    choose_file = files_in_dir[random.randint(0, len(files_in_dir)-1)]
    print('reading from file:', choose_file)
    load_vid = cv2.VideoCapture(os.path.join(parent_dir,choose_file))
    totalFrames = load_vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total frames: ', totalFrames)
    randomFrameNumber = random.randint(0, totalFrames)
    print('getting frame: ', randomFrameNumber)
    load_vid.set(cv2.CAP_PROP_POS_FRAMES, randomFrameNumber)
    ret, frame = load_vid.read()
    print(frame)
    if ret == True:
        print('got frame plotting it')
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        if real:
            write_path = real_writepath
            print('real image, writing to real path')
        else:
            write_path = fake_writepath
        write_to = os.path.join(write_path,choose_file+'frame'+str(randomFrameNumber)+'.jpg')
        if os.path.isfile(write_to):
            print('duplicate frame, skipping write')
            return(0)
        print('writing frame for: ', choose_file)
        cv2.imwrite(write_to, frame)
        print('successful frame write to: ', write_to)
        # # Saves image of the current frame in jpg file
        # name = './data/frame' + str(currentFrame) + '.jpg'
        # print ('Creating...' + name)
        # cv2.imwrite(name, frame)
    else:
        print('missed frame')
    load_vid.release()
    return(1)

# test call
# sample_frame(real_path)
# print('saved frame')

#%% sample real frames
# sampling 10000 frames
count = 0
print('starting_file_count: ', len(os.listdir(real_writepath)))
while len(os.listdir(real_writepath)) < 5000: # for sample in range(10000):
    count += sample_frame(real_path)
    print('real frames written: ', count)


#%% sample fake frames
# sample 1000 frames per category
count = 0
print('starting_file_count: ', len(os.listdir(fake_writepath)))
while len(os.listdir(fake_writepath)) < 5000: # for sample in range(10000):
    random_fake_subdir = fake_dirs[random.randint(0, len(fake_dirs) - 1)]
    print('selecting from: ', random_fake_subdir)
    path_to_sample_from = os.path.join(fake_path,random_fake_subdir)
    count += sample_frame(path_to_sample_from, real=False)
    print('fake frames written: ', count)


