import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

#data isnt always in its final processed form
#we use transforms to perform manipulation to make it good

#all torchvision datasets has 2 params 
    #transform to modify features
    #target_transform to modify labels

#fashionmnist features are in pil, and labels are ints. to train we need features as tensors, labels as one hot ecoded tensors. 

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10,dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
)