'''
this script is for the training code of Project 2. It should be similar as the one in Project 1.

-------------------------------------------
INTRO:
You should write your codes or modify codes between the 
two '#####' lines. The codes between two '=======' lines 
are used for logging or printing, please keep them unchanged 
in your submission. 

You need to debug and run your codes in somewhere else (e.g. Jupyter 
Notebook). This file is only used for the evaluation stage and for
your submission and marks. Please make sure the codes are running 
smoothly using your trained model.

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

from network import Network # the network you used

# training process. 
def train_net(net, trainloader, valloader):
########## ToDo: Your codes goes below #######
    val_accuracy = 0
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    return val_accuracy
##############################################

############################################
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

train_image_path = '../train/' 
validation_image_path = '../validation/' 

trainset = ImageFolder(train_image_path, train_transform)
valset = ImageFolder(validation_image_path, train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'. 
# DO NOT CHANGE THIS PART.

network = Network()
if args.cuda:
    network = network.cuda()

# train and eval your trained network
# you have to define your own 
val_acc = train_net(network, trainloader, valloader)

print("final validation accuracy:", val_acc)

# ==================================