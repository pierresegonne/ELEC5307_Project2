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
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network # the network you used

def manual_plateau(epoch):
    if epoch >= 32:
        return 0.5**2
    if epoch >= 16:
        return 0.5
    else:
        return 1

# training process.
def train_net(net, train_loader, validation_loader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
########## ToDo: Your codes goes below #######
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001) # regularisation tecnique to reduce overfitting 
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, [manual_plateau])
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 2
    nMiniBatches = 20

    net.train()

    training_losses = []
    epochs = []
    batches = []
    validation_losses = []
    validation_accuracies = []
    training_losses = []
    training_accuracies = []

    running_loss_training = 0.0
    total_training = 0
    correct_training = 0

    running_validation_loss = 0
    correct_validation = 0
    total = 0


    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss_training = 0.0
        scheduler.step()

        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_training += loss.item()

            # find the training accuracy
            _, predicted_training = torch.max(outputs.data, 1)
            total_training += labels.size(0)
            correct_training += (predicted_training == labels).sum().item()


            if i % nMiniBatches == nMiniBatches-1:    # print every n mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss_training / nMiniBatches))

                # find the training loss
                training_losses.append(running_loss_training/nMiniBatches)
                epochs.append(epoch+1)
                batches.append(i+1 + epoch*len(train_loader.dataset)/BATCH_SIZE)

                # find the training accuracy
                training_accuracy = 100 * float(correct_training) / total_training
                training_accuracies.append(training_accuracy)

                running_loss_training = 0.0
                total_training = 0
                correct_training = 0

                running_validation_loss = 0
                correct_validation = 0
                total = 0

                # calulate the training loss and validation periodically
                for i, validation_data in enumerate(validation_loader, 0):
                    with torch.no_grad():
                        net.eval()
                        # get the inputs
                        inputs, validation_labels = validation_data
                        inputs, validation_labels = inputs.to(device),validation_labels.to(device)

                        # calculate validation loss
                        validation_outputs = net(inputs)
                        validation_loss = criterion(validation_outputs, validation_labels)
                        running_validation_loss += validation_loss.item()

                        # calcualte validation accuracy
                        _, predicted = torch.max(validation_outputs.data, 1)
                        total += validation_labels.size(0)
                        correct_validation += (predicted == validation_labels).sum().item()

                validation_losses.append(running_validation_loss/len(validation_loader))
                validation_accuracy = 100 * float(correct_validation) / total
                validation_accuracies.append(validation_accuracy)

    print('Finished Training')
    return validation_accuracies[-1]

##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop, Resize and any other transform you think is useful. 
# Remember to make the normalize value same as in the training transformation.

NORMALIZATION = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION['mean'], std=NORMALIZATION['std']),
])

validation_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION['mean'], std=NORMALIZATION['std']),
])

####################################
BATCH_SIZE = 32
####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

train_image_path = 'data_resized/Task1/train/'
validation_image_path = 'data_resized/Task1/val/'

import os
print(os.listdir('.'))

trainset = ImageFolder(train_image_path, train_transform)
valset = ImageFolder(validation_image_path, validation_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'.
# DO NOT CHANGE THIS PART.

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network().to(device)


# train and eval your trained network
# you have to define your own
val_acc = train_net(network, trainloader, valloader)

print("final validation accuracy:", val_acc)

# ==================================