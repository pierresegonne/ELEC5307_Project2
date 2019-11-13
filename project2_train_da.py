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

import copy
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import random
import umap

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder



from network_da import Network_DA # the network you used

def lr_annealing(original_learning_rate, p):
  ALPHA = 10
  BETA = 0.75
  return original_learning_rate / ((1 + (ALPHA * p))**BETA)

def compute_lambda(p):
  GAMMA = 10
  return (2 / (1 + math.exp(-GAMMA * p))) - 1

def evaluate(model, loader, device, flag_source):
    # Eval flag
    model.eval()

    # Initialize metrics
    loss = 0.
    accuracy = 0.
    accuracy_domain = 0.

    # Criterion
    criterion = nn.CrossEntropyLoss() # From paper

    with torch.no_grad():
        for (inputs, labels) in loader:

            # To device
            inputs = inputs.to(device)
            if flag_source == True:
                labels = labels.to(device)

            # Prepare domain labels
            if flag_source == True:
                domains = torch.zeros(len(inputs)).long().to(device)
            else:
                domains = torch.ones(len(inputs)).long().to(device)

            label_outputs, domain_outputs = model(inputs, lbd=0)

            # Update loss
            if flag_source == True:
                loss += criterion(label_outputs, labels).data.item()

            # Update Accuracies
            predicted_labels = label_outputs.data.max(1)[1]
            if flag_source == True:
                accuracy += predicted_labels.eq(labels.data).sum().item()
            predicted_domains = domain_outputs.data.max(1)[1]
            accuracy_domain += predicted_domains.eq(domains.data).sum().item()

    # Average metrics
    loss /= len(loader)
    accuracy /= len(loader.dataset)
    accuracy_domain /= len(loader.dataset)

    return loss, accuracy, accuracy_domain

# Training process. You can add validation loader and manually label some images for validation if you want to see the target performance.
def train_net(model, source_loader, target_loader,
    source_only=False, evaluate_model_every=1,
    save_model_every=50, model_filename='dann.pth',
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
########## ToDo: Your codes goes below #######
    # In task 2, you cannot use the labels in the target loader. As the result, you need to use domain adaptation techniques
    #                                             to reduce the domain gaps and achieve better performance for target domain.

    # For example 1. Gradient Reversal Layers (Search github!)
    #             2. Train with Target Pseudo Labels (Define new dataset and dataloader, you might need to create your own
    #                                                 instead of direct 'import ImageFolder')
    #             3. Others. Eg. MMD

    val_accuracy, nbr_accuracy = 0, 0
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    # Optimizer and criterion
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    print('==== Start Training ====')

      # Recordings
    number_epochs = 2
    for epoch in range(number_epochs):
        # Train flag
        model.train()

        # Mixed loader
        loader = zip(source_loader, target_loader)
        len_loader = min(len(source_loader), len(target_loader)) # Avoid doing list(zip(...)) for performance

        for iteration, ((source_inputs, source_labels), (target_inputs, _)) in enumerate(loader):

            # Training progress p
            p = (epoch * len_loader + iteration) / (number_epochs * len_loader) # progress, linearly from 0 to 1
            lbd = compute_lambda(p)
            adjusted_lr = lr_annealing(learning_rate, p)
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lr

            # To device
            source_inputs = source_inputs.to(device)
            source_labels = source_labels.to(device)
            target_inputs = target_inputs.to(device)

            # Prepare domain labels
            source_domains = torch.zeros(len(source_inputs)).long().to(device)
            target_domains = torch.ones(len(target_inputs)).long().to(device)

            # Reset grad
            optimizer.zero_grad()

            # Source domain
            source_label_outputs, source_domain_outputs = model(source_inputs, lbd=lbd)
            source_labels_loss = criterion(source_label_outputs, source_labels)
            source_domains_loss = criterion(source_domain_outputs, source_domains)

            # Target domain
            _, target_domain_outputs = model(target_inputs, lbd=lbd)
            target_domains_loss = criterion(target_domain_outputs, target_domains)

            # Combine losses
            if source_only:
                loss = source_labels_loss
            else:
                loss = source_labels_loss + source_domains_loss + target_domains_loss

            # Optimize
            loss.backward()
            optimizer.step()

            # Evaluate
            if ((epoch + 1) % evaluate_model_every == 0):
                source_loss, source_accuracy, source_accuracy_domain = evaluate(model, source_loader, device, True)
                target_loss, target_accuracy, target_accuracy_domain = evaluate(model, target_loader, device, False)

                # For submission
                nbr_accuracy += 1
                val_accuracy += source_accuracy

            # Save temporary model
            if ((epoch + 1) % save_model_every == 0):
                torch.save(model.state_dict(), model_filename)

    # Save final model
    torch.save(model.state_dict(), model_filename)

    print('\n==== END TRAINING ====')

    val_accuracy /= nbr_accuracy

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

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def make_dataset(dir, extensions=None, is_valid_file=None):
    """
        Adaption of https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                images.append(path)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class TargetImageFolder(Dataset):

    def __init__(self, root, loader, extensions=None, transform=None):
        self.extensions = extensions
        self.loader = loader
        self.samples = make_dataset(root, extensions=IMG_EXTENSIONS)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, []

source_image_path = './data_resized/Task2/source/'
target_image_path = './data_resized/Task2/target/'

SOURCE_NORMALIZATION = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
TARGET_NORMALIZATION = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

batch_size_source = 128 // 2 # From paper
batch_size_target = 128 // 2

source_transform = transforms.Compose([
                                       transforms.RandomResizedCrop(227),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=SOURCE_NORMALIZATION['mean'], std=SOURCE_NORMALIZATION['std']),
])

target_transform = transforms.Compose([
                                       transforms.Resize((227, 227)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=SOURCE_NORMALIZATION['mean'], std=SOURCE_NORMALIZATION['std']),
])

sourceset = ImageFolder(source_image_path, source_transform)
targetset = TargetImageFolder(target_image_path, default_loader, transform=target_transform, extensions=IMG_EXTENSIONS)

sourceloader = torch.utils.data.DataLoader(sourceset, batch_size=batch_size_source,
                                         shuffle=True, num_workers=2)
targetloader = torch.utils.data.DataLoader(targetset, batch_size=batch_size_target,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'.
# DO NOT CHANGE THIS PART.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network_DA(number_classes=len(sourceset.classes)).to(device)

# train and eval your trained network
# you have to define your own
val_acc = train_net(network, sourceloader, targetloader)

print("final validation accuracy:", val_acc)

# ==================================