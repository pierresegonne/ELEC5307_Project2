"""
This file will take all the pictures from task 1 data folder and
resize them so that they all have acceptable size.
"""
import cv2
import glob
import os


DATA_FOLDER = './data/Task2' # Change for different task
DATA_RESIZED_FOLDER = './data_resized/Task2' # Change for different task
SUBFOLDERS = ['Source', 'Target'] # Change for different task

all_pictures = []
for subfolder in SUBFOLDERS:
    if subfolder != 'Target':
        category_folders = glob.glob(f'{DATA_FOLDER}/{subfolder}/*')
        for category_folder in category_folders:
            pictures = glob.glob(f'{category_folder}/*')
            all_pictures += pictures
    else: # Target pictures are unlabeled
        pictures = glob.glob(f'{DATA_FOLDER}/{subfolder}/*')
        all_pictures += pictures

for index, picture in enumerate(all_pictures):
    img = cv2.imread(picture)
    if img.shape[0] * img.shape[1] > 1000 * 1000:
        # Make the longest size 1000
        resolution_ratio = (1000*1000)/(img.shape[0]*img.shape[1])
        new_height, new_width = int(img.shape[0] * resolution_ratio), int(img.shape[1] * resolution_ratio)
        img = cv2.resize(img, (new_width, new_height))

    new_picture = '{}/{}'.format(DATA_RESIZED_FOLDER, '/'.join(picture.split('/')[3:]))
    new_picture_directory = '/'.join(new_picture.split('/')[:-1])
    if not os.path.exists(new_picture_directory):
        os.makedirs(new_picture_directory)
    cv2.imwrite(new_picture, img)

    if (index + 1) % 100 == 0:
        print(f' --- Treated {index + 1}/{len(all_pictures)} pictures')


