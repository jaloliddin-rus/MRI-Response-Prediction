# -*- coding: utf-8 -*-
"""
Created on 24/02/2024

@author: Zahiriddin & Jaloliddin
"""
# Utilities
import os
import numpy as np
import pandas as pd
import torch
import monai
import pickle
import time
import argparse
from tqdm import tqdm


# Data Loader
from monai.data import Dataset, DataLoader, ITKReader
from torch.utils.data import random_split, Subset

# Model Import
from CustomRegressor import CustomRegressor
from DenseNet import DenseNet121, DenseNet169, DenseNet201
from Autoencoder import AutoEncoder
from ViTAutoEnc import ViTAutoEnc
from BasicUNet import BasicUNet
from EfficientNetBN import EfficientNetBN

# Optimizer
from torch.optim import Adam
import torch.optim as optim

# Loss Functions
from torch.nn import L1Loss, MSELoss

# Image Reader
import SimpleITK as sitk

# Plotting Graphs
import matplotlib.pyplot as plt

# Transformations
from scipy.ndimage import rotate
from monai.transforms import LoadImage, Compose, ToTensor, RandRotate90
import torch.nn.functional as F

print("is CUDA enabled?: ", torch.cuda.is_available())

# Pearson Correlation Loss
def pearson_correlation_loss(outputs, labels):
    outputs_mean = outputs - outputs.mean()
    labels_mean = labels - labels.mean()
    corr = (outputs_mean * labels_mean).sum() / torch.sqrt((outputs_mean ** 2).sum() * (labels_mean ** 2).sum())
    return 1 - corr

# Cosine Similarity Loss
def cosine_similarity_loss(outputs, labels):
    # Use PyTorch's functional API to compute cosine similarity
    cosine_sim = F.cosine_similarity(outputs, labels, dim=1)
    # Since cosine similarity returns a value between -1 and 1, where 1 means totally similar,
    # we subtract from 1 to get a loss which needs to be minimized.
    return 1 - cosine_sim.mean()

def load_3d_tiff(tiff_path):
    """Load a 3D TIFF file and return as a numpy array."""
    image = sitk.ReadImage(tiff_path)
    return sitk.GetArrayFromImage(image)
    
def select_model(architecture, spatial_dims, in_channels, out_channels, **kwargs):
    if architecture == 'CustomRegressor':
        return CustomRegressor(**kwargs)
    elif architecture == 'DenseNet121':
        return DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, dropout_prob = 0.2, **kwargs)
    elif architecture == 'DenseNet169':
        return DenseNet169(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, **kwargs)
    elif architecture == 'DenseNet201':
        return DenseNet201(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, **kwargs)
    elif architecture == 'AutoEncoder':
        return AutoEncoder(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, channels = (16, 32, 64, 128, 256), strides = (2, 2, 2, 2, 2), **kwargs)
    elif architecture == 'ViTAutoEnc':
        return ViTAutoEnc(in_channels=in_channels, img_size = (64, 64, 64), patch_size= (16, 16, 16), out_features=out_channels, **kwargs)
    elif architecture == 'BasicUNet':
        return BasicUNet(spatial_dims = spatial_dims, in_channels=in_channels, out_channels=out_channels, **kwargs)
    elif architecture == 'EfficientNetB0':
        return EfficientNetBN(model_name = 'efficientnet-b0', spatial_dims = spatial_dims, in_channels = in_channels, num_output_features = out_channels, **kwargs)
    elif architecture == 'EfficientNetB3':
        return EfficientNetBN(model_name = 'efficientnet-b3', spatial_dims = spatial_dims, in_channels = in_channels, num_output_features = out_channels, **kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
        
def parse_args():
    parser = argparse.ArgumentParser(description="Architecture and training parameters")
    parser.add_argument('--architecture', type=str, default='CustomRegressor',
                        help='Model architecture: CustomRegressor, DenseNet121, DenseNet169, DenseNet201, etc.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--loss_fn', type=str, default='MSELoss',
                    help='Loss function to use: L1Loss, MSELoss, CustomL1Loss')
    parser.add_argument('--dir', type=str, default='data/temp',
                    help='Directory of Data (structured: data/animal/...)')

    args = parser.parse_args()
    return args

args = parse_args()

data_dir = args.dir
data_list = []

# First loop: Iterate through main folders (animal MRIs)
for animal_dir in os.listdir(data_dir):
    animal_path = os.path.join(data_dir, animal_dir)
    
    # Check if it's a directory (just an additional check)
    if os.path.isdir(animal_path):
        
        # Second loop: Iterate through each chunk folder inside the animal folder
        for chunk_dir in os.listdir(animal_path):
            chunk_path = os.path.join(animal_path, chunk_dir)
            
            tiff_files = [os.path.join(chunk_path, f) for f in os.listdir(chunk_path) if f.endswith('.tiff')]
            npy_file = os.path.join(chunk_path, "signal.npy")
            
            if len(tiff_files) == 9:
                data_list.append({"images": tiff_files, "label": npy_file})

# Define custom collate function to handle multiple tiff inputs
def custom_collate(batch):
    images = [item['images'] for item in batch]
    labels = [item['label'] for item in batch]
    # return [torch.stack(images, dim=0), torch.stack(labels, dim=0)]
    return {'images': torch.stack(images, dim=0), 'label': torch.stack(labels, dim=0)}

def custom_l1_loss(outputs, labels, alpha=0.5, beta=0.25, gamma=0.25):
    loss_L1 = L1Loss()(outputs, labels)
    loss_pearson = pearson_correlation_loss(outputs, labels)
    loss_cosine = cosine_similarity_loss(outputs, labels)
    # Combine losses
    loss = alpha * loss_L1 + beta * loss_pearson + gamma * loss_cosine
    return loss

class TiffDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load each 3D volume
        volumes = [load_3d_tiff(img) for img in self.data[index]['images']]
        
        # Apply transformations if any
        if self.transform:
            volumes = [self.transform(vol) for vol in volumes]
    
        # Convert each volume to a PyTorch tensor
        #volumes = [torch.tensor(vol, dtype=torch.float32) for vol in volumes]
        volumes = [vol.clone().detach() for vol in volumes] # If you do not need to calculate gradients for volumes
    
        # Stack the volumes along a new dimension
        volume_4d = torch.stack(volumes, dim=0)  # This will have shape [9, 64, 64, 64]
        
        label = np.load(self.data[index]['label'])
        return {"images": volume_4d, "label": torch.tensor(label, dtype=torch.float32)}

class RandomRotate3D:
    def __init__(self, degrees):
        """
        Initialize random rotation for 3D volumes.

        :param degrees: Max rotation in degrees. Rotation angle will be randomly sampled from (-degrees, degrees).
        """
        self.degrees = degrees

    def __call__(self, volume):
        """
        Rotate the 3D volume.

        :param volume: 3D numpy array
        :return: Rotated 3D numpy array
        """
        # Randomly choose a rotation angle.
        angle_x = np.random.uniform(-self.degrees, self.degrees)
        angle_y = np.random.uniform(-self.degrees, self.degrees)
        angle_z = np.random.uniform(-self.degrees, self.degrees)

        # Rotate along each axis separately.
        volume_rotated = rotate(volume, angle_x, axes=(0, 1), reshape=False, mode='nearest')
        volume_rotated = rotate(volume_rotated, angle_y, axes=(0, 2), reshape=False, mode='nearest')
        volume_rotated = rotate(volume_rotated, angle_z, axes=(1, 2), reshape=False, mode='nearest')

        return volume_rotated

min_angle = -np.pi / 24
max_angle = np.pi / 24

rotate_transform = RandomRotate3D(degrees=90)

loader = LoadImage(reader=ITKReader())
train_transforms = Compose([
    #LoadImage(loader, image_only=True),  # Load tiff image
    #RandRotate(90),
    #RandRotated(keys=["image", "label"], prob=1, range_x=(30, 70), range_y=(30, 70)),
    #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # Horizontal flip
    #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # Vertical flip
    #rotate_transform,
    RandRotate90(prob=0.5, max_k=3),
    ToTensor()
])

test_transforms = Compose([
    ToTensor()
])

# for sample in data_list[:3]:  # Checking the first 10 for brevity
#     volume = load_3d_tiff(sample['images'][0])
#     print("Loaded volume shape:", volume.shape)

# train_ds = TiffDataset(data_list, train_transforms)

# batch_size = 64  # or any other desired batch size
# num_workers = 8
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)

# # Validation & Training Set
# train_size = int(0.8 * len(data_list))
# val_size = len(data_list) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(train_ds, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)

train_ds = TiffDataset(data_list, train_transforms)
test_ds = TiffDataset(data_list, test_transforms)

# Paths to the saved indices
train_indices_path = 'train_indices.pkl'
val_indices_path = 'val_indices.pkl'
test_indices_path = 'test_indices.pkl'

# Check if train indices file exists
if os.path.exists(train_indices_path) and os.path.exists(val_indices_path) and os.path.exists(test_indices_path):
    # Load indices
    with open(train_indices_path, 'rb') as f:
        train_indices = pickle.load(f)
    with open(val_indices_path, 'rb') as f:
        val_indices = pickle.load(f)
    with open(test_indices_path, 'rb') as f:
        test_indices = pickle.load(f)
else:
    # Split the data and save indices
    train_size = int(0.8 * len(data_list))
    test_size = len(data_list) - train_size
    train_data, test_data = random_split(data_list, [train_size, test_size])

    val_size = int(0.1 * len(train_data))
    train_size = len(train_data) - val_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    with open(train_indices_path, 'wb') as f:
        pickle.dump(train_dataset.indices, f)
    with open(val_indices_path, 'wb') as f:
        pickle.dump(val_dataset.indices, f)
    with open(test_indices_path, 'wb') as f:
        pickle.dump(test_data.indices, f)
        
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_data.indices


# Create datasets using the loaded/created indices
train_dataset = Subset(train_ds, train_indices)
val_dataset = Subset(test_ds, val_indices)
test_dataset = Subset(test_ds, test_indices)

num_train_samples = len(train_dataset)
num_val_samples = len(val_dataset)
num_test_samples = len(test_dataset)

print(f"Number of training samples: {num_train_samples}")
print(f"Number of validation samples: {num_val_samples}")
print(f"Number of testing samples: {num_test_samples}")

#exit(0)

# Create DataLoaders
batch_size = 32  # or your preferred batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)

# for i in range(5):  # Check the first 5 items
#     sample = train_ds[i]
#     print(f"{i}. {type(sample['images'])} - {type(sample['label'])}")

meta_tensor_sample = monai.data.meta_tensor.MetaTensor(torch.tensor([1, 2, 3]), meta_data={"test": 123})
converted_tensor = torch.tensor(meta_tensor_sample.numpy()) if isinstance(meta_tensor_sample, monai.data.meta_tensor.MetaTensor) else meta_tensor_sample
#print(type(converted_tensor))

model_parameters = {
    'CustomRegressor': {
        'constructor': CustomRegressor,  # Directly reference the constructor for models without parameters
    },
    'DenseNet121': {
        'constructor': select_model,
        'params': {'spatial_dims': 3, 'in_channels': 9, 'out_channels': 50, 'pretrained': False}
    },
    'DenseNet169': {
        'constructor': select_model,
        'params': {'spatial_dims': 3, 'in_channels': 9, 'out_channels': 50, 'pretrained': False}
    },
    'DenseNet201': {
        'constructor': select_model,
        'params': {'spatial_dims': 3, 'in_channels': 9, 'out_channels': 50, 'pretrained': False}
    },
    'AutoEncoder': {
        'constructor': select_model,
        'params': {'spatial_dims': 3, 'in_channels': 9, 'out_channels': 50}
    },
    'ViTAutoEnc': {
        'constructor': select_model,
        'params': {'spatial_dims': 3, 'in_channels': 9, 'out_channels': 50}
    },
    'BasicUNet':{
        'constructor': select_model,
        'params': {'spatial_dims': 3, 'in_channels': 9, 'out_channels': 50}
    },
    'EfficientNetB0':{
        'constructor': select_model,
        'params': {'spatial_dims': 3, 'in_channels': 9, 'out_channels': 50}
    },
    'EfficientNetB3':{
        'constructor': select_model,
        'params': {'spatial_dims': 3, 'in_channels': 9, 'out_channels': 50}
    }
}

def get_model_by_architecture(architecture):
    if architecture in model_parameters:
        model_info = model_parameters[architecture]
        # Check if the model uses select_model as its constructor
        if model_info['constructor'] == select_model:
            # Ensure architecture is passed as a parameter for select_model
            return model_info['constructor'](architecture, **model_info['params'])
        else:
            # For other constructors that do not require the architecture as an argument
            return model_info['constructor']()
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

if __name__ == "__main__":

    model = get_model_by_architecture(args.architecture)

    # Select the loss function based on user input
    if args.loss_fn == "L1Loss":
        loss_function = L1Loss()
    elif args.loss_fn == "MSELoss":
        loss_function = MSELoss()
    elif args.loss_fn == "CustomL1Loss":
        def loss_function(outputs, labels):
            return custom_l1_loss(outputs, labels, alpha=0.5, beta=0.25, gamma=0.25)
    else:
        raise ValueError("Unknown loss function specified")

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Add a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 300
    train_losses = []
    val_losses = []

    # Early stopping parameters
    patience = 25
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    start_time = time.time()  # Start time
    epoch_data_list = [] # Initialize an empty list to collect epoch data

    for epoch in range(args.epochs):
        epoch_time = time.time()  # Start time of epoch
        if early_stop:
            print("Early stopping due to no improvement in validation loss!")
            break
        #print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            inputs, labels = batch_data['images'], batch_data['label']
            # print(inputs.size())
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
             for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                inputs, labels = batch_data['images'], batch_data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        epoch_end_time = time.time()  # End time of epoch
        epoch_time = (epoch_end_time - epoch_time) / 60  # Calculate epoch training time
        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Epoch Time (min): {epoch_time:.2f}")
        
        epoch_data = {
            'Epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_val_loss,
            'Epoch Time (min)': epoch_time
        }
        epoch_data_list.append(epoch_data)

        # Check early stopping conditions
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the model with the best validation loss
            torch.save(model.state_dict(), f'models/best_{args.architecture}_{args.loss_fn}.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                early_stop = True
        
        # Step the scheduler
        scheduler.step()

    end_time = time.time()  # End time
    training_time = (end_time - start_time) / 60  # Calculate total training time
    print(f"Training completed in: {training_time:.2f} minutes")

    # Plotting the training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig(f'figures/{args.architecture}_{args.loss_fn}.svg', format='svg')
    plt.savefig(f'figures/{args.architecture}_{args.loss_fn}.png', format='png', dpi=1000)
    plt.show()

    df = pd.DataFrame(epoch_data_list)
    df.to_csv(f'training/{args.architecture}_{args.loss_fn}.csv', index=False)

    exit(0)