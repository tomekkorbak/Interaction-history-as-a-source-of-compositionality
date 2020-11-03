import torch
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


colors = ['blue', 'cyan', 'gray', 'green', 'magenta']
object_types = ['box', 'sphere', 'cylinder', 'torus', 'ellipsoid']


class ColoredFiguresDataset(ImageFolder):

    def __getitem__(self, index):
        filename, _ = self.imgs[index]
        color, figure = filename.split('/')[-2].split('-')
        color_idx, figure_idx = colors.index(color), object_types.index(figure)
        label = torch.LongTensor([color_idx, figure_idx])
        return super(ColoredFiguresDataset, self).__getitem__(index)[0], label


def prepare_datasets():
    train_path = 'train'
    test_path = 'test'

    train_dataset = ColoredFiguresDataset(root=train_path, transform=ToTensor())
    test_dataset = ColoredFiguresDataset(root=test_path, transform=ToTensor())
    full_dataset = data.ConcatDataset([train_dataset, test_dataset])
    return full_dataset, train_dataset, test_dataset
