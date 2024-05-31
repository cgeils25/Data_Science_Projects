import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image


class ImmuneCellImageDataset(Dataset):
    """
    Custom Pytorch dataset for immune cell images.
    Expects class to be the second to last directory in the image path.
    """
    def __init__(self, img_paths: list, class_map: dict, transform=None, device=torch.device('cpu')):
        self.img_paths=img_paths # list of image paths
        self.class_map=class_map # dictionary mapping class to tensor label
        self.transform=transform # torchvision v2.Compose object
        self.device = device # device to move tensors to
        
    def __getitem__(self, idx):
        """Retrieves image and label at given index. Applies transformation to image.

        Args:
            idx (int): index of image to retrieve

        Returns:
            img_as_tensor (torch.Tensor): image as tensor (with transformations applied)
            img_label (torch.Tensor): label as tensor
        """
        # get image path with index
        img_path = self.img_paths[idx]

        # read image as tensor
        img_as_tensor = read_image(img_path)
        assert self.transform is not None, "Error: must specify image transformation"
        img_as_tensor = self.transform(img_as_tensor)  # apply specified transformation to image
        img_as_tensor = img_as_tensor.to(self.device)  # move to given device
        
        # get image label
        img_class = img_path.split('/')[-2]  # image class: str
        img_label = self.class_map[img_class]  # maps class to tensor label
        img_label = img_label.to(self.device)

        return img_as_tensor, img_label

    def __len__(self):
        return len(self.img_paths)

    def show_image(self, idx):
        """
        for ipython notebook use
        shows image at given idx  ** without transformations
        """
        
        # get image path with index
        img_path = self.img_paths[idx]

        # get image class
        img_class = img_path.split('/')[-2]
        
        # show image
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f'Cell Type: {img_class}, Idx: {idx}')

    def immune_cell_counts(self):
        """
        Returns a dictionary of cell type counts in the dataset
        """
        cell_type_counts = {cell_type: 0 for cell_type, _ in self.class_map.items()}  # getting cell types from class map because I'm lazy

        for img_path in self.img_paths:
            img_cell_type = img_path.split('/')[-2]
            cell_type_counts[img_cell_type] += 1

        return cell_type_counts