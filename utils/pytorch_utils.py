import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import copy


class ImmuneCellImageDataset(Dataset):
    def __init__(self, img_paths: list, class_map: dict, transform=None, device=torch.device('cpu')):
        """
        Pytorch Dataset object for immune cell image data.
        Should be pretty easy to extend this to other image datasets ¯\_(ツ)_/¯
        Args:
            img_paths (list): _description_
            class_map (dict): _description_
            transform (torchvision.transforms.v2.Compose): torchvision Compose object, specifying image transformations. Defaults to None.
            device (torch.device, optional): pytorch device to load data onto. Defaults to torch.device('cpu').
        """
        self.img_paths=img_paths
        self.class_map=class_map # maps class names to tensor labels
        self.transform=transform # specifies image transformations
        self.device = device
        self.unique_labels=list(class_map.values()) 
        
    def __getitem__(self, idx):
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
        shows image at given idx without transformations
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
        Get counts of each immune cell type in the dataset, return as dict
        """
        cell_type_counts = {cell_type: 0 for cell_type in self.class_map.items()}  # getting cell types from class map because I'm lazy

        for img_path in self.img_paths:
            img_cell_type = img_path.split('/')[-2]
            cell_type_counts[img_cell_type] += 1

        return cell_type_counts
    

def classification_accuracy(dataloader, model):
    """Compute the proportion of correct predictions for a given model on a given dataset.
    Outputs are considered correct if the model's highest output index matches the target label.

    Args:
        dataloader (torch.utils.data.DataLoader): pytorch dataloader object
        model (torch.nn.Module): pytorch model object

    Returns:
        p_correct (float): proportion of correct predictions
    """
    # set model to evaluation mode
    model.eval()

    total_correct = 0
    total_samples = len(dataloader.dataset)

    for batch in tqdm(iter(dataloader)):
        # Every data instance is an input + label pair
        batch_inputs, batch_labels = batch

        # Make predictions for this batch
        batch_outputs = model(batch_inputs).detach()
        
        # Obtaining model consensus
        batch_consensus = torch.argmax(batch_outputs, dim=1)
        total_correct += sum(batch_consensus == batch_labels).item() # get num targets that match consensus

    p_correct = total_correct / total_samples

    return p_correct


def validate_classification_model(dataloader, model, loss_fn, device):
    """Validate a classification model on a given dataset, return both mean loss and class-specific losses

    Args:
        dataloader (torch.utils.data.DataLoader): a pytorch dataloader object. Must have a dataset attribute with unique_labels: list(torch.Tensor)
        model (torch.nn.Module): a pytorch model object
        loss_fn (torch.optim): a pytorch loss function object
        device (torch.device): a pytorch device object

    Returns:
        mean_validation_loss (float): mean loss across all batches
        losses_for_unique_labels (dict): dictionary of losses for each unique label
    """
    # set model to evaluation mode
    model.eval()
    
    batch_loss = 0
    batch_loss_list = []

    unique_labels = dataloader.dataset.unique_labels
    losses_for_unique_labels_raw = {unique_label: [] for unique_label in unique_labels}
    
    for batch in tqdm(iter(dataloader)):
        # Every data instance is an input + label pair
        batch_inputs, batch_labels = batch

        # Make predictions for this batch
        batch_outputs = model(batch_inputs).detach()
        
        # Compute the loss
        loss = loss_fn(batch_outputs, batch_labels)

        # get loss for each label
        for unique_label in unique_labels:
            if any(batch_labels == unique_label):
                batch_idxs_for_unique_label = batch_labels == unique_label
                
                batch_outputs_for_unique_label = batch_outputs[batch_idxs_for_unique_label]
                num_samples_in_batch = len(batch_outputs_for_unique_label)
                batch_targets_for_unique_label = unique_label.repeat(1, num_samples_in_batch).flatten().to(device)
               
                loss_for_unique_label = loss_fn(batch_outputs_for_unique_label, batch_targets_for_unique_label)
                loss_for_unique_label = loss_for_unique_label.item()
            else:
                loss_for_unique_label = 0
                num_samples_in_batch = 0
                
            losses_for_unique_labels_raw[unique_label].append((loss_for_unique_label, num_samples_in_batch))
        
        # Extract batch loss as float
        batch_loss = loss.item()

        # add batch loss to running list
        batch_loss_list.append(batch_loss) 

    losses_for_unique_labels = {unique_label: [] for unique_label in unique_labels}

    # obtain losses for each class by aggregating individual batch losses
    for unique_label in unique_labels:
        loss_list_for_unique_label_raw = losses_for_unique_labels_raw[unique_label]
        total_samples = sum([pair[1] for pair in loss_list_for_unique_label_raw])
        loss_weights = [pair[1]/total_samples for pair in loss_list_for_unique_label_raw]
        assert sum(loss_weights) == 1 
        weighted_losses = [pair[0] * loss_weights[i] for i, pair in enumerate(loss_list_for_unique_label_raw)]
        aggregated_class_loss = sum(weighted_losses)
        losses_for_unique_labels[unique_label] = aggregated_class_loss

    mean_validation_loss = np.mean(batch_loss_list)
    return mean_validation_loss, losses_for_unique_labels


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    """Train a PyTorch model for one epoch on a given dataloader

    Args:
        dataloader (torch.utils.data.DataLoader): a pytorch dataloader object
        model (torch.nn.Module): a pytorch model object
        loss_fn (torch.optim): a pytorch loss function object
        optimizer (torch.optim): a pytorch optimizer object
        device (torch.device): a pytorch device object

    Returns:
        last_loss (float): loss for the last batch
    """
    
    # set model to train mode
    model.train()
    
    running_loss = 0.
    last_loss = 0.
    
    for batch_idx, batch in enumerate(iter(dataloader)):
        # Every data instance is an input + label pair
        batch_inputs, batch_labels = batch

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        batch_outputs = model(batch_inputs)

        # Compute the loss and its gradients
        loss = loss_fn(batch_outputs, batch_labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        num_batches_per_update = 50  # number of batches for each print statement
        
        if batch_idx % num_batches_per_update == 0 and not batch_idx == 0:
            last_loss = running_loss / num_batches_per_update # loss per batch
            print(f'Batch {batch_idx} loss: {last_loss}')
            running_loss = 0.

    return last_loss