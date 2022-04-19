import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

DATA_PATH = 'data'

_transform_train = transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(), 
          transforms.Normalize(mean = [.5], std = [.5])
          ])

_transform_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean = [.5], std = [.5])
            ])

_fMNIST_train_dataset = datasets.FashionMNIST(
    DATA_PATH, 
    train=True, 
    transform=_transform_train, 
    download=True
)

_fMNIST_test_dataset = datasets.FashionMNIST(
    DATA_PATH, 
    train=False, 
    transform=_transform_test, 
    download=True)

fmnist_train_loader = torch.utils.data.DataLoader(
    dataset=_fMNIST_train_dataset,
    batch_size=512, 
    shuffle=True
)

fmnist_test_loader = torch.utils.data.DataLoader(
    dataset=_fMNIST_test_dataset,
    batch_size=512, 
    shuffle=False
)