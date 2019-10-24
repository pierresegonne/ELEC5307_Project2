'''
this script is for the training code of Project 2. It should be similar as the one in Project 1.

-------------------------------------------
INTRO:
You should write your codes or modify codes between the 
two '#####' lines. The codes between two '=======' lines 
are used for logging or printing, please keep them unchanged 
in your submission. 

-------------------------------------------
USAGE:
In your final update, please keep the file name as 'python2_test.py'.

>> python project2_test.py
This will run the program on CPU to test on your trained nets for the Fruit test dataset in Task 1.

>> python project2_test.py --cuda
This will run the program on GPU to test on your trained nets for the Fruit test dataset in Task 1. 
You can ignore this if you do not have GPU or CUDA installed.

>> python project2_test.py --da
This will run the program on CPU to test your domain adaptive model for the Office target dataset in Task 2.

>> python project2_test.py --da --cuda
This will run the program on GPU to test your domain adaptive model for the Office target dataset in Task 2.

-------------------------------------------
NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email: wzha8158@uni.sydney.edu.au, dzho8854@uni.sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network_da import Network_DA # the network you used

# Training process. You can add validation loader and manually label some images for validation if you want to see the target performance.
def train_net(net, sourceloader, targetloader): 
########## ToDo: Your codes goes below #######
    # In task 2, you cannot use the labels in the target loader. As the result, you need to use domain adaptation techniques
    #                                             to reduce the domain gaps and achieve better performance for target domain.
    
    # For example 1. Gradient Reversal Layers (Search github!)
    #             2. Train with Target Pseudo Labels (Define new dataset and dataloader, you might need to create your own 
    #                                                 instead of direct 'import ImageFolder')
    #             3. Others. Eg. MMD

    val_accuracy = 0
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    return val_accuracy

##############################################

# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop, Resize and any other transform you think is useful. 
# Remember to make the normalize value same as in the training transformation.

train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

####################################

####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

source_image_path = '../source/' 
target_image_path = '../target/' 

sourceset = ImageFolder(source_image_path, train_transform)
targetset = ImageFolder(target_image_path, train_transform)

sourceloader = torch.utils.data.DataLoader(sourceset, batch_size=4,
                                         shuffle=True, num_workers=2)
targetloader = torch.utils.data.DataLoader(targetset, batch_size=4,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'. 
# DO NOT CHANGE THIS PART.

network = Network_DA()
if args.cuda:
    network = network.cuda()

# train and eval your trained network
# you have to define your own 
val_acc = train_net(network, sourceloader, targetloader)

print("final validation accuracy:", val_acc)

# ==================================