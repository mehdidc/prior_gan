import os

import torchvision.datasets as dset
import torchvision.transforms as transforms

from utils import Invert
from utils import Gray

DATA_PATH = '/home/mcherti/work/data'
def data_path(folder):
    return os.path.join(DATA_PATH, folder)


def load_dataset(dataset_name, split='full'):
    if dataset_name == 'mnist':
        dataset = dset.MNIST(
            root=data_path('mnist'), 
            download=True,
            transform=transforms.Compose([
                transforms.Scale(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        return dataset
    elif dataset_name == 'fonts':
        dataset = dset.ImageFolder(root=data_path('fonts/' + split),
          transform=transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            Invert(),
            Gray(),
            transforms.Normalize((0.5,), (0.5,)),
         ]))
        return dataset
    elif dataset_name == 'coco':
        dataset = dset.ImageFolder(root=data_path('coco'),
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'coco_256':
        dataset = dset.ImageFolder(root=data_path('coco'),
            transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'footwear':
        dataset = dset.ImageFolder(root=data_path('shoes/ut-zap50k-images'),
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'celeba':
        dataset = dset.ImageFolder(root=data_path('celeba'),
            transform=transforms.Compose([
            transforms.Scale(78),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'birds':
        dataset = dset.ImageFolder(root=data_path('birdsfull'),
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    else:
        raise ValueError('Error')
